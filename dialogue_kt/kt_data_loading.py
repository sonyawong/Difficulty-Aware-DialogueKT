from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from dialogue_kt.utils import device
from dialogue_kt.models.dkt_sem import ALT_ARCH
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt, kt_user_prompt_ordinal, dkt_sem_prompt, ORDINAL_ABILITY_INSTRUCTION, ORDINAL_DIFFICULTY_INSTRUCTION
import logging

log_path = "./logs/data_loading.log"

logging.basicConfig(
    filename=log_path,  # 指定日志文件名
    filemode="w",  # 追加模式，可以是'a'或'w'，默认为'a'
    format="%(asctime)s [%(levelname)s] [%(funcName)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def apply_annotations(sample: dict, apply_na: bool = True):
    dialogue = sample["dialogue"]
    anno = sample["annotation"]
    if "error" in anno:
        return None
    # Handle dialogues beginning with turn 0 (student-initiated)
    if dialogue[0]["turn"] == 0:
        anno["turn 0"] = {"correct": None, "kcs": []}
    # Copy correctness and kcs into dialogue
    for dia_turn in dialogue:
        anno_turn = anno[f"turn {dia_turn['turn']}"]
        corr = anno_turn["correct"]
        kcs = anno_turn["kcs"]
        if apply_na:
            corr = None if not kcs else corr
            kcs = [] if corr is None else kcs
        dia_turn["correct"] = dia_turn["og_correct"] = corr
        dia_turn["kcs"] = kcs
    # Use human annotation of correctness for final turn
    if dialogue[-1]["kcs"]: # Skip if no KCs for final turn since correct must be None
        if "expected_result" in sample["meta_data"]: # CoMTA
            dialogue[-1]["correct"] = sample["meta_data"]["expected_result"] == "Answer Accepted"
        elif "self_correctness" in sample["meta_data"]: # MathDial
            if dialogue[-1]["correct"] is not None: # Final turn could be closing remarks, so skip if not tagged as having correctness
                if sample["meta_data"]["self_correctness"] == "Yes":
                    dialogue[-1]["correct"] = True
                elif sample["meta_data"]["self_correctness"] == "Yes, but I had to reveal the answer":
                    dialogue[-1]["correct"] = None
                elif sample["meta_data"]["self_correctness"] == "No":
                    dialogue[-1]["correct"] = False
    return dialogue

class DatasetBase(Dataset):
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class LMKTDatasetUnpacked(DatasetBase):
    def __init__(self, data: pd.DataFrame, tokenizer, args, skip_first_turn: bool = False):
        self.data = []
        failed = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            is_first_turn = True
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                # Skip first tagged turn at test time for fairness with baselines
                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue
                self.data.append({
                    "dialogue_idx": idx,
                    "prompts": [
                        tokenizer.apply_chat_template([
                            {"role": "system", "content": kt_system_prompt(args)},
                            {"role": "user", "content": kt_user_prompt(sample, dialogue, turn["turn"], kc, args)},
                            {"role": "assistant", "content": f"\n"} # Newline would precede True or False prediction
                        ], tokenize=False)
                        for kc in turn["kcs"]
                    ],
                    "label": turn["correct"],
                    "kcs": turn["kcs"]
                })
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of dialogues: {len(self.data)}")
        print_label_stats(self.data, "LMKTDatasetUnpacked")

class LMKTCollatorUnpacked:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        all_prompts = [prompt for sample in batch for prompt in sample["prompts"]]
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)
        return {
            "input_ids": prompts_tokenized.input_ids,
            "attention_mask": prompts_tokenized.attention_mask,
            "last_idxs": prompts_tokenized.attention_mask.sum(dim=-1) - 2, # Take index of token before eos
            "num_kcs": torch.LongTensor([len(sample["prompts"]) for sample in batch]).to(device),
            "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
            "meta_data": batch
        }

class LMKTDatasetPacked(DatasetBase):
    def __init__(self, data: pd.DataFrame, tokenizer, args, skip_first_turn: bool = False):
        self.data = []
        failed = 0
        skipped_too_long = []  # 记录超长序列
        
        # Get max_length from tokenizer
        max_length = getattr(tokenizer, "model_max_length", 131072)  # Llama 3.2 default
        if max_length > 100000:  # If unlimited or very large, set a reasonable limit
            max_length = 8192  # Common Llama 3 context length
        
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            is_first_turn = True
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                # Skip first tagged turn at test time for fairness with baselines
                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue
                # Create base prompt followed by continuation blocks
                # Ordinal IRT: KC text inline in user message + 2 level blocks (ability, difficulty)
                prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": kt_system_prompt(args)},
                    {"role": "user", "content": kt_user_prompt_ordinal(sample, dialogue, turn["turn"], turn["kcs"], args)},
                ], tokenize=False)
                level_block_contents = [
                    ORDINAL_ABILITY_INSTRUCTION,
                    ORDINAL_DIFFICULTY_INSTRUCTION
                ]
                level_conts = []
                for block_content in level_block_contents:
                    cont = tokenizer.apply_chat_template([
                        {"role": "user", "content": block_content},
                        {"role": "assistant", "content": "\n"}
                    ], tokenize=False)
                    level_conts.append(" " + cont.split("user<|end_header_id|>\n\n")[1])
                prompt = prompt + "".join(level_conts)
                
                # Check prompt length BEFORE adding to dataset
                tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                prompt_length = tokenized.input_ids.shape[1]
                
                if prompt_length > max_length:
                    # Record this sample as too long
                    skipped_info = {
                        "original_dataframe_index": idx,
                        "turn_number": turn["turn"],
                        "prompt_length": prompt_length,
                        "max_length": max_length,
                        "num_kcs": len(turn["kcs"]),
                        "kcs": str(turn["kcs"]),
                        "qid": sample.get("qid", "N/A"),
                        "scenario": sample.get("scenario", "N/A"),
                    }
                    skipped_too_long.append(skipped_info)
                    logging.warning(
                        f"Skipping dialogue {idx}, turn {turn['turn']}: "
                        f"length {prompt_length} > max {max_length}"
                    )
                    continue  # Skip this sample
                
                # logging.info(f"Processed dialogue, prompts:{prompt}, correctness: {turn['correct']}, kcs: {turn['kcs']}")
                
                self.data.append({
                    "dialogue_idx": idx,
                    "prompt": prompt,
                    "label": turn["correct"],
                    "kcs": turn["kcs"]
                })
        print(f"{failed} / {len(data)} dialogues failed processing")
        total_valid = len(self.data)
        total_skipped = len(skipped_too_long)
        print(f"Samples within max_length ({max_length}): {total_valid}  |  Skipped (too long): {total_skipped}")
        
        # Report and save skipped samples
        if skipped_too_long:
            print(f"⚠️  Skipped {total_skipped} samples due to exceeding max_length ({max_length})")
            
            # Save to CSV for analysis
            skipped_df = pd.DataFrame(skipped_too_long)
            skipped_file = f"./logs/skipped_too_long_{args.dataset}_{args.tag_src}.csv"
            skipped_df.to_csv(skipped_file, index=False)
            print(f"   Saved details to: {skipped_file}")
            print(f"   Length stats: min={skipped_df['prompt_length'].min()}, "
                  f"max={skipped_df['prompt_length'].max()}, "
                  f"mean={skipped_df['prompt_length'].mean():.1f}")
        
        print_label_stats(self.data, "LMKTDatasetPacked")

class LMKTCollatorPacked:
    def __init__(self, tokenizer, use_irt: bool = False, kc_dict: dict = None):
        self.tokenizer = tokenizer
        self.use_irt = use_irt
        self.kc_dict = kc_dict

        if use_irt:
            self.ability_token_ids = torch.LongTensor([
                tokenizer("GOOD", add_special_tokens=False).input_ids[-1],
                tokenizer("BAD", add_special_tokens=False).input_ids[-1],
            ])
            self.difficulty_token_ids = torch.LongTensor([
                tokenizer("HARD", add_special_tokens=False).input_ids[-1],
                tokenizer("EASY", add_special_tokens=False).input_ids[-1],
            ])
            self.end_dialogue_token_id = tokenizer.convert_tokens_to_ids('[END DIALOGUE]')
            print(f"Ordinal IRT mode: 2 level blocks (ability + difficulty), 2 classes each")
            print(f"  Ability token IDs: GOOD={self.ability_token_ids[0]}, BAD={self.ability_token_ids[1]}")
            print(f"  Difficulty token IDs: HARD={self.difficulty_token_ids[0]}, EASY={self.difficulty_token_ids[1]}")
            print(f"  [END DIALOGUE] token ID: {self.end_dialogue_token_id}")



    def __call__(self, batch):
        prompts = [sample["prompt"] for sample in batch]
        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = prompts_tokenized.input_ids.to(device)
        batch_size, max_seq_len = input_ids.shape
        eos_idxs = [
            (input_ids[seq_idx] == self.tokenizer.eos_token_id).nonzero().squeeze().cpu()
            for seq_idx in range(batch_size)
        ]
        
        end_dialogue_idxs = None
        kc_ids = None

        if self.use_irt:
            end_dialogue_idxs = []
            for seq_idx in range(batch_size):
                dialogue_pos = (input_ids[seq_idx] == self.end_dialogue_token_id).nonzero().squeeze()
                if dialogue_pos.dim() == 0 and dialogue_pos.numel() > 0:
                    end_dialogue_idxs.append(dialogue_pos.item())
                else:
                    if dialogue_pos.numel() == 0:
                        print(f"Warning: [END DIALOGUE] token not found, Sequence index: {seq_idx}")
                    end_dialogue_idxs.append(dialogue_pos[0].item() if dialogue_pos.numel() > 0 else 0)
            end_dialogue_idxs = torch.LongTensor(end_dialogue_idxs)

            kc_ids_list = []
            for sample in batch:
                sample_kc_ids = [self.kc_dict[kc]+1 for kc in sample["kcs"]]
                kc_ids_list.append(torch.LongTensor(sample_kc_ids))
            kc_ids = pad_sequence(kc_ids_list, batch_first=True, padding_value=0)
        
        # Create default lower triangular 3D attention mask
        attention_mask = torch.ones((max_seq_len, max_seq_len)).tril().repeat(batch_size, 1, 1)
        tril_mask = attention_mask[0].type(torch.bool)
        # Create default 2D position id matrix
        position_ids = torch.arange(max_seq_len).repeat(batch_size, 1)
        # Set attention mask and position ids for each sequence
        for seq_idx in range(batch_size):
            # Get end of context
            context_end_idx = eos_idxs[seq_idx][1]
            # Initialize to no attention to any tokens after context
            attention_mask[seq_idx, :, position_ids[seq_idx] >= context_end_idx] = 0
            # Update attention mask and position ids for each KC
            start_idx = context_end_idx + 1
            for end_idx in eos_idxs[seq_idx][3::2]:
                # Set position ids as if KC immediately followed context
                position_ids[seq_idx, start_idx : end_idx + 1] = torch.arange(context_end_idx, context_end_idx + end_idx - start_idx + 1)
                # Set KC attention mask to lower triangular to permit self-attention
                cur_tril_mask = tril_mask.clone()
                cur_tril_mask[end_idx + 1:] = False
                cur_tril_mask[:, :start_idx] = False
                attention_mask[seq_idx, cur_tril_mask] = 1
                # Go to next KC
                start_idx = end_idx + 1

        # Ordinal IRT: restrict ability block to only attend up to [END DIALOGUE]
        if self.use_irt:
            for seq_idx in range(batch_size):
                ed_idx = end_dialogue_idxs[seq_idx].item()
                ctx_end = eos_idxs[seq_idx][1].item()

                # Ability block = first continuation block: context_end+1 .. eos[3]
                ability_start = ctx_end + 1
                ability_end = eos_idxs[seq_idx][3].item()

                # Zero out ability block's attention to context tokens after [END DIALOGUE]
                if ed_idx + 1 < ctx_end:
                    attention_mask[seq_idx, ability_start:ability_end + 1, ed_idx + 1:ctx_end] = 0

                # Adjust position_ids so ability block follows [END DIALOGUE] directly
                block_len = ability_end - ability_start + 1
                position_ids[seq_idx, ability_start:ability_end + 1] = torch.arange(
                    ed_idx + 1, ed_idx + 1 + block_len
                )

        # Get index of token before eos for all prediction tasks (KCs and/or Level Blocks)
        all_last_idxs = [idxs[3::2] - 1 for idxs in eos_idxs]
        
        if self.use_irt:
            # level_last_idxs: the last 2 positions (ability block, difficulty block)
            level_last_idxs = torch.stack([lasts[-2:] for lasts in all_last_idxs]).to(device)
            last_idxs = pad_sequence(all_last_idxs, batch_first=True)
        else:
            level_last_idxs = None
            last_idxs = pad_sequence(all_last_idxs, batch_first=True)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask.unsqueeze(1).to(device), # Add singleton head dimension
            "position_ids": position_ids.to(device),
            "last_idxs": last_idxs,
            "num_kcs": torch.LongTensor([len(sample["kcs"]) for sample in batch]).to(device),
            "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
            "meta_data": batch
        }
        
        if self.use_irt:
            result["end_dialogue_idx"] = end_dialogue_idxs.to(device)
            result["level_last_idxs"] = level_last_idxs
            result["ability_token_ids"] = self.ability_token_ids.to(device)
            result["difficulty_token_ids"] = self.difficulty_token_ids.to(device)
            result["kc_ids"] = kc_ids.to(device)

        return result


class DKTDataset(DatasetBase):
    def __init__(self, data: pd.DataFrame, kc_dict: Dict[str, int], kc_emb_matrix: torch.Tensor, sbert_model: SentenceTransformer):
        self.data = []
        failed = 0
        num_data_points = 0
        num_correct = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            dialogue_data = {
                "labels": [], "labels_flat": [], "kc_ids": [], "kc_ids_flat": [], "turn_end_idxs": [],
                "teacher_turns": [], "student_turns": [], "kcs": [], "kc_embs": [],
                "dialogue": dialogue, "dialogue_idx": idx
            }
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                dialogue_data["labels"].append(turn["correct"])
                dialogue_data["kc_ids"].append([kc_dict[kc] for kc in turn["kcs"]])
                for kc in turn["kcs"]:
                    dialogue_data["labels_flat"].append(turn["correct"])
                    dialogue_data["kc_ids_flat"].append(kc_dict[kc])
                dialogue_data["turn_end_idxs"].append(len(dialogue_data["kc_ids_flat"]) - 1)
                dialogue_data["teacher_turns"].append(turn["teacher"])
                dialogue_data["student_turns"].append(turn["student"])
                dialogue_data["kcs"].append(turn["kcs"])
                if kc_emb_matrix is not None:
                    dialogue_data["kc_embs"].append(
                        kc_emb_matrix[dialogue_data["kc_ids"][-1]].mean(dim=0)
                    )
            # Add dialogue if at least 2 turns tagged, otherwise nothing to predict
            if len(dialogue_data["labels"]) > 1:
                self.data.append(dialogue_data)
                num_data_points += len(dialogue_data["labels"])
                num_correct += sum(dialogue_data["labels"])
        # Batch encode all dialogue text
        if sbert_model is not None:
            batch_size = 512
            if ALT_ARCH:
                seqs = [dkt_sem_prompt(tt, st, kcs, corr)
                        for dialogue in self.data
                        for tt, st, kcs, corr in zip(dialogue["teacher_turns"], dialogue["student_turns"], dialogue["kcs"], dialogue["labels"])]
                result_embs = []
                for batch_start_idx in range(0, len(seqs), batch_size):
                    batch = seqs[batch_start_idx : batch_start_idx + batch_size]
                    result_embs.append(sbert_model.encode(batch, convert_to_tensor=True))
                result_embs = torch.concat(result_embs, dim=0)
                turn_counter = 0
                for dialogue in self.data:
                    seq_len = len(dialogue["labels"])
                    dialogue["turn_embs"] = result_embs[turn_counter : turn_counter + seq_len]
                    turn_counter += seq_len
            else:
                seqs = [turn for dialogue in self.data for turn in dialogue["teacher_turns"]] + [
                        turn for dialogue in self.data for turn in dialogue["student_turns"]]
                result_embs = []
                for batch_start_idx in range(0, len(seqs), batch_size):
                    batch = seqs[batch_start_idx : batch_start_idx + batch_size]
                    result_embs.append(sbert_model.encode(batch, convert_to_tensor=True))
                result_embs = torch.concat(result_embs, dim=0)
                turn_counter = 0
                for dialogue in self.data:
                    seq_len = len(dialogue["labels"])
                    dialogue["teacher_embs"] = result_embs[turn_counter : turn_counter + seq_len]
                    stud_start = result_embs.shape[0] // 2
                    dialogue["student_embs"] = result_embs[stud_start + turn_counter : stud_start + turn_counter + seq_len]
                    turn_counter += seq_len
        self.majority_class = 1 if num_correct / num_data_points >= .5 else 0
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Num dialogues: {len(self.data)}, num data points: {num_data_points}, {num_correct} correct")

class DKTCollator:
    def __init__(self, flatten_kcs: bool):
        self.flatten_kcs = flatten_kcs

    def __call__(self, batch):
        labels = pad_sequence(
            [torch.LongTensor(seq["labels"]) for seq in batch],
            batch_first=True, padding_value=-100 # Pad with -100 to ignore loss on padding regions
        )
        # # Fill in KC ids, 2D matrix (length x max num KCs) per sequence
        num_kcs = pad_sequence(
            [torch.LongTensor([len(kc_ids) for kc_ids in seq["kc_ids"]]) for seq in batch],
            batch_first=True, padding_value=1 # Pad with 1 to avoid division by 0
        )
        max_num_kcs = num_kcs.max()
        kc_ids = torch.zeros((*num_kcs.shape, max_num_kcs), dtype=torch.long)
        for seq_idx, seq in enumerate(batch):
            for turn_idx, turn_kc_ids in enumerate(seq["kc_ids"]):
                kc_ids[seq_idx, turn_idx, :len(turn_kc_ids)] = torch.LongTensor(turn_kc_ids)

        result = {
            "labels": labels.to(device),
            "kc_ids": kc_ids.to(device),
            "num_kcs": num_kcs.to(device)
        }

        if self.flatten_kcs:
            # Add flattened versions of KC ids and labels for unrolled model input
            kc_ids_flat = pad_sequence([torch.LongTensor(seq["kc_ids_flat"]) for seq in batch], batch_first=True)
            labels_flat = pad_sequence([torch.LongTensor(seq["labels_flat"]) for seq in batch], batch_first=True)
            turn_end_idxs = pad_sequence([torch.LongTensor(seq["turn_end_idxs"]) for seq in batch], batch_first=True)
            result = {
                **result,
                "labels_flat": labels_flat.to(device),
                "kc_ids_flat": kc_ids_flat.to(device),
                "turn_end_idxs": turn_end_idxs.to(device)
            }

        # Add text embeddings for DKT-Sem
        if batch[0]["kc_embs"] and not ALT_ARCH:
            kc_embs = pad_sequence([torch.stack(seq["kc_embs"]) for seq in batch], batch_first=True)
            teacher_embs = pad_sequence([seq["teacher_embs"] for seq in batch], batch_first=True)
            student_embs = pad_sequence([seq["student_embs"] for seq in batch], batch_first=True)
            result = {
                **result,
                "kc_embs": kc_embs,
                "teacher_embs": teacher_embs,
                "student_embs": student_embs
            }
        elif "turn_embs" in batch[0]:
            turn_embs = pad_sequence([seq["turn_embs"] for seq in batch], batch_first=True)
            result = {
                **result,
                "turn_embs": turn_embs
            }

        return result

def get_dataloader(dataset: Dataset, collator, batch_size: int, shuffle: bool):
    return DataLoader(dataset, collate_fn=collator, batch_size=batch_size, shuffle=shuffle)
    # g = torch.Generator()
    # g.manual_seed(221)
    # return DataLoader(
    #     dataset, collate_fn=collator, batch_size=batch_size, shuffle=shuffle,
    #     worker_init_fn=seed_worker, generator=g
    # )

def print_label_stats(data_list, dataset_name="Dataset"):
    """Print label distribution statistics"""
    if not data_list:
        return
    
    labels = [d["label"] for d in data_list]
    label_0_count = sum(1 for l in labels if l == 0.0)
    label_1_count = sum(1 for l in labels if l == 1.0)
    total = len(labels)
    
    print(f"{dataset_name} - Label distribution:")
    print(f"  Incorrect (0): {label_0_count} ({100*label_0_count/total:.1f}%)")
    print(f"  Correct (1): {label_1_count} ({100*label_1_count/total:.1f}%)")
    
    if label_0_count == 0:
        print(f"  ⚠️  WARNING: All labels are 1 (Correct) - no negative samples!")
    if label_1_count == 0:
        print(f"  ⚠️  WARNING: All labels are 0 (Incorrect) - no positive samples!")

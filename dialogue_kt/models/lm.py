import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from dialogue_kt.utils import get_checkpoint_path

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

def get_base_model(base_model_name: str, tokenizer: AutoTokenizer, quantize: bool):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=bnb_config if quantize else None,
        # f32 seems helpful for train/test time consistency when quantizing, bf16 performs best for non-quantized
        torch_dtype=torch.float32 if quantize else torch.bfloat16,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    return base_model

def get_model(base_model_name: str, test: bool,
              model_name: str = None, pt_model_name: str = None,
              r: int = None, lora_alpha: int = None,
              quantize: bool = True, use_gradient_checkpointing: bool = True,
              use_irt: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.bos_token

    if use_irt:
        num_added = tokenizer.add_special_tokens({
            'additional_special_tokens': ['[BEGIN DIALOGUE]', '[END DIALOGUE]']
        })
        print(f"Added {num_added} special tokens (IRT ordinal)")
        for token in ['[BEGIN DIALOGUE]', '[END DIALOGUE]']:
            print(f"  {token} token ID: {tokenizer.convert_tokens_to_ids(token)}")

    base_model = get_base_model(base_model_name, tokenizer, quantize)

    if use_irt:
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to {len(tokenizer)} tokens")

    if test and model_name:
        print("Initializing inference-time model from fine-tuned LoRA adapters")
        base_model = PeftModel.from_pretrained(base_model, get_checkpoint_path(model_name))
    elif not test:
        if quantize:
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=use_gradient_checkpointing)
        else:
            base_model.gradient_checkpointing_enable()
            base_model.enable_input_require_grads()
        if pt_model_name:
            print("Initializing trainable model from pre-trained LoRA adapters")
            base_model = PeftModel.from_pretrained(base_model, get_checkpoint_path(pt_model_name), is_trainable=True, adapter_name="default")
        else:
            print("Initializing trainable model with new LoRA adapters")
            peft_config = LoraConfig(
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
            base_model = get_peft_model(base_model, peft_config)
    else:
        print("Initializing inference-time model from pre-trained weights")

    if use_irt:
        from dialogue_kt.models.lm_kc_emb import LMKTWithKCEmbedding
        model = LMKTWithKCEmbedding(base_model)
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        model.logit_scale.data = model.logit_scale.data.to(device=device, dtype=dtype)
        if test and model_name:
            try:
                model.load_mlp_head(get_checkpoint_path(model_name))
                print("Loaded IRT head from checkpoint")
            except FileNotFoundError:
                print("Warning: IRT head not found in checkpoint, using random initialization")
        return model, tokenizer
    else:
        return base_model, tokenizer

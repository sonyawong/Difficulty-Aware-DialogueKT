from typing import List, Optional
from transformers import AutoTokenizer

from dialogue_kt.data_loading import correct_to_str, standards_to_str

# ===== General functions =====

def get_dialogue_text(dialogue: List[dict], turn_idx: int = None, include_labels: bool = False, tag_wrapper: bool = True):
    lines = []
    
    target_turn_idx = turn_idx
    if turn_idx is not None:
        target_turn = next((t for t in dialogue if t["turn"] == turn_idx), None) # equal to [x for x in dialogue if x["turn"] == turn_idx]
        if target_turn and (target_turn.get("teacher") == "N/A" or not target_turn.get("teacher")) and target_turn.get("student") and target_turn.get("student") != "N/A":

            for i in range(turn_idx - 1, 0, -1):
                teacher_turn = next((t for t in dialogue if t["turn"] == i), None)
                if teacher_turn and teacher_turn.get("teacher") and teacher_turn.get("teacher") != "N/A":
                    target_turn_idx = i
                    break
    
    for turn in dialogue:
        if "teacher" in turn and turn["teacher"] and turn["teacher"] != "N/A":
            if target_turn_idx is not None and target_turn_idx == turn["turn"]:
                lines.append(f"[BEGIN CURRENT TEACHER TURN]\nTeacher Turn {turn['turn']}: {turn['teacher']}\n[END CURRENT TEACHER TURN]")
            else:
                lines.append(f"Teacher Turn {turn['turn']}: {turn['teacher']}")
        
        if turn_idx is not None and turn_idx == turn["turn"]:
            break
        if "student" in turn and turn["student"] and turn["student"] != "N/A":
            lines.append(f"Student Turn {turn['turn']}: {turn['student']}")
        if include_labels: 
            lines.append(f"Student Turn {turn['turn']} Correct: {correct_to_str(turn['correct'])}")
            lines.append(f"Turn {turn['turn']} Knowledge Components: {standards_to_str(turn['kcs'], ' ')}")
    prompt = "\n".join(lines[:-1])

    if tag_wrapper:
        prompt = "[BEGIN DIALOGUE]\n" + prompt + "\n[END DIALOGUE]\n\n\n" + f"{lines[-1]}\n" 
    return prompt

def get_eedi_context(sample: dict, init_answer: bool = False):
    context = f"[BEGIN QUESTION]\n{sample['meta_data']['question'].strip()}\n[END QUESTION]\n"
    if init_answer:
        context += f"\n\n[BEGIN INIT ANSWER]\nInitial Student Answer: Incorrect\n[END INIT ANSWER]\n" # init answer is always incorrect
    return context

def get_mathdial_context(sample: dict, init_answer: bool = False):
    context = (f"[BEGIN PROBLEM]\n{sample['meta_data']['question'].strip()}\n[END PROBLEM]\n\n"
               f"[BEGIN CORRECT SOLUTION]\n{sample['meta_data']['correct_solution'].strip()}\n[END CORRECT SOLUTION]\n\n"
               f"[BEGIN INCORRECT STUDENT SOLUTION]\n{sample['meta_data']['incorrect_solution'].strip()}\n[END INCORRECT STUDENT SOLUTION]")
    if init_answer:
        context += f"\n\n[BEGIN INIT ANSWER]\nInitial Student Answer: Incorrect\n[END INIT ANSWER]\n" # init answer is always incorrect
    return context

# ===== Annotation prompting =====
def get_dataset_desc(args):
    if args.dataset == "eedi":
        return EEDI_DESC
    if args.dataset == "mathdial":
        return MATHDIAL_DIALOGUE_DESC
    raise Exception(f"No dataset description defined for {args.dataset}")

EEDI_DESC = "the student is working through a multiple-choice diagnostic math question. You are given the question and the corresponding correct answer"
MATHDIAL_DIALOGUE_DESC = "the student is attempting to solve a math problem. You are also given this problem, its correct solution, and the incorrect solution the student initially gave."


def get_true_false_tokens(tokenizer: AutoTokenizer):
    true = tokenizer("True").input_ids[-1]
    false = tokenizer("False").input_ids[-1]
    return true, false

# ===== KT model prompting =====

KT_SYSTEM_PROMPT = """You are an experienced math teacher. Given a dialogue where {desc}, determine from prior context whether the student has the knowledge needed to answer the teacher’s next question, and respond only “True” or “False”."""

ORDINAL_ABILITY_INSTRUCTION = """ Based on the student's responses and demonstrated understanding in the dialogue, classify the student's current ability level. Respond with exactly one token from: GOOD FAIL."""

ORDINAL_DIFFICULTY_INSTRUCTION = """ Based on the question content and required knowledge components, classify the question's difficulty level. Respond with exactly one token from: HARD EASY."""

def kt_system_prompt(args):
    return KT_SYSTEM_PROMPT.format(desc=get_dataset_desc(args))

def kt_user_prompt(sample: dict, dialogue_anno: List[dict], turn_idx: int, kc: Optional[str], args):
    prompt = ""
    if args.dataset == "eedi":
        prompt += get_eedi_context(sample, init_answer=args.prompt_inc_init_answer) + "\n\n"
    prompt += get_dialogue_text(dialogue_anno, turn_idx=turn_idx, include_labels=args.prompt_inc_labels)
    return prompt

def kt_user_prompt_ordinal(sample: dict, dialogue_anno: List[dict], turn_idx: int, kcs: List[str], args):
    """User prompt for ordinal IRT - includes KC text inline in the user message."""
    prompt = ""
    if args.dataset == "eedi":
        prompt += get_eedi_context(sample, init_answer=args.prompt_inc_init_answer) + "\n\n"
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample, init_answer=args.prompt_inc_init_answer) + "\n\n"
    prompt += get_dialogue_text(dialogue_anno, turn_idx=turn_idx, include_labels=args.prompt_inc_labels)
    if kcs:
        prompt += "\n\n[BEGIN KC]\n" + "\n".join(f"{kc}" for kc in kcs) + "\n[END KC]\n"
    return prompt

def dkt_sem_prompt(teacher_turn: str, student_turn: str, kcs: List[str], correct: bool):
    return f"""A teacher and a student are having a dialogue about math concepts. Below is a single turn pair from that dialouge, where the student responds to the teacher. In addition, there are knowledge components that represent the learning objectives in the teacher's question. Finally, it is identified if the student's response to the teacher was correct or incorrect.
Teacher: {teacher_turn}
Student: {student_turn}
Knowledge Components: {standards_to_str(kcs, ' ')}
Student Correct: {correct_to_str(correct)}"""

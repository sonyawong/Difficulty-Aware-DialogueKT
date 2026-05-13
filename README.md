# Difficulty-Aware Dialogue Knowledge Tracing

This is the repo for **Difficulty-Aware-DialogueKT** in COMPSCI 682: An Interpretable LLM-Based Ability and Difficulty Modeling Framework for Knowledge Tracing in Tutoring Dialogues.

---

## Installation

```bash
pip install -r requirements.txt
```

**Hardware:** Experiments use a single NVIDIA L40 GPU. The default base model is `meta-llama/Meta-Llama-3.1-8B-Instruct`.

---

## Train

```bash
sh run.sh
```

or directly:

```bash
python -m dialogue_kt.main train \
    --dataset eedi \
    --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model_type lmkt \
    --model_name my_model \
    --tag_src atc \
    --use_irt True \
    --lr 2e-4 \
    --epoch 5 \
    --use_best_auc True
```

---

## Evaluate

```bash
sh evaluate.sh
```

or directly:

```bash
python -m dialogue_kt.main test \
    --dataset eedi \
    --crossval \
    --model_type lmkt \
    --model_name my_model \
    --tag_src atc \
    --base_model meta-llama/Meta-Llama-3.1-8B-Instruct
```


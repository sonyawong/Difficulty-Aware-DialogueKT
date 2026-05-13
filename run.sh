
# HF_HUB_OFFLINE=1

# module load cuda/12.1
# module load conda/latest

# conda activate /work/pi_andrewlan_umass_edu/shuang_umass_edu/.conda/envs/llmkt_work
# cd /work/pi_andrewlan_umass_edu/shuang_umass_edu/Difficulty-Aware-DialogueKT


python -m dialogue_kt.main train \
    --dataset mathdial \
    --base_model /work/pi_andrewlan_umass_edu/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ \
    --model_type lmkt \
    --model_name lmkt_mathdial_8b \
    --tag_src atc \
    --use_irt True \
    --batch_size 1 \
    --fold 1 \
    --lr 2e-4 \
    --epoch 5 \
    --use_best_auc True \
    --wandb_project dialogue-kt-2026 \
    --debug True
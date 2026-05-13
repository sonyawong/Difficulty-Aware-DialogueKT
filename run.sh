

python -m dialogue_kt.main train \
    --dataset eedi \
    --base_model model_path \
    --model_type lmkt \
    --model_name lmkt_eedi_8b \
    --tag_src mathdial_format \
    --use_irt True \
    --batch_size 1 \
    --fold 1 \
    --lr 2e-4 \
    --epoch 5 \
    --use_best_auc True \
    --wandb_project dialogue-kt-2026 \
    --debug True
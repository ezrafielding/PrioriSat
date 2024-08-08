#! /bin/bash

python -m open_clip_train.main \
    --val-data="../datasets/NWPU-Captions/test.csv"  \
    --model RN50 \
    --pretrained ./logs/RN50_RS_FineTuned_50epochs/checkpoints/epoch_50.pt \
    --csv-img-key filepath \
    --csv-caption-key caption

python -m open_clip_train.main \
    --val-data="../datasets/NWPU-Captions/test.csv"  \
    --model ViT-B-16 \
    --pretrained ./logs/ViT-B-16_RS_FineTuned_50epochs/checkpoints/epoch_50.pt \
    --csv-img-key filepath \
    --csv-caption-key caption
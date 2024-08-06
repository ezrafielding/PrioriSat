#! /bin/bash

python -m open_clip_train.main \
    --val-data="../datasets/NWPU-Captions/test.csv"  \
    --model RN50 \
    --pretrained ./logs/RN50_RS_FineTuned/checkpoints/epoch_30.pt \
    --csv-img-key filepath \
    --csv-caption-key caption \
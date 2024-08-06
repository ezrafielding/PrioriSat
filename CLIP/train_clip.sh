#! /bin/bash

python -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --train-data="../datasets/NWPU-Captions/train.csv"  \
    --val-data="../datasets/NWPU-Captions/val.csv"  \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=50 \
    --workers=8 \
    --model ViT-B-16 \
    --pretrained datacomp_xl_s13b_b90k

python -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --train-data="../datasets/NWPU-Captions/train.csv"  \
    --val-data="../datasets/NWPU-Captions/val.csv"  \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=50 \
    --workers=8 \
    --model RN50 \
    --pretrained openai

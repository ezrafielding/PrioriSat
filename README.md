# PrioriSat
[![GitHub](https://img.shields.io/github/license/ezrafielding/PrioriSat)](https://github.com/ezrafielding/PrioriSat/blob/main/LICENSE)

## Abstract
Nanosatellites increasingly produce more data than can be feasibly downlinked due to their limited bandwidth and power. To address this, on-board systems that prioritize scientifically significant data for downlink using artificial intelligence are essential. This work compares two approaches that utilize state-of-the-art natural language processing techniques to prioritize remote sensing (RS) image data. The first approach uses YOLOv8 and Llama2 to extract image features and compare them with text descriptions via cosine similarity, while the second employs a fine-tuned CLIP model for this task. Both methods were evaluated on the VERTECS Camera Control Board which features a Raspberry Pi Compute Module 4 (CM4). The CLIP approach, particularly the ResNet50-based model, showed superior performance in prioritizing and sequencing RS image downlinks, proving more efficient despite higher computational demands. This study demonstrates that on-orbit prioritization using natural language descriptions is viable and can enhance the scientific output of CubeSat missions.

## Oriented Bounding Box (OBB) Approach
Uses fine-tuned YOLOv8 and Llama2 models as image and text feature extractors, respectively.

Two YOLOv8 models with different image input sizes were trained (960 px and 1280 px).

## CLIP Approach
Uses CLIP fine-tuned for RS images.

ResNet50 and ViT-B-16 -based CLIP models were trained.

## Datasets
- DOTA-v1.5: https://captain-whu.github.io/DOTA/index.html
- NWPU-Captions: https://ieeexplore.ieee.org/document/9866055


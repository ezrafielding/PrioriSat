# PrioriSat: Flexible Natural Language-Based Image Data Downlink Prioritization for Nanosatellites
[![GitHub](https://img.shields.io/github/license/ezrafielding/PrioriSat)](https://github.com/ezrafielding/PrioriSat/blob/main/LICENSE) [![DOI](https://img.shields.io/badge/DOI-10.3390%2Faerospace11110888-blue)](https://doi.org/10.3390/aerospace11110888)

## Abstract
Nanosatellites increasingly produce more data than can be downlinked within a reasonable time due to their limited bandwidth and power. Therefore, an on-board system is required to prioritize scientifically significant data for downlinking, as described by scientists. This paper determines whether natural language processing can be used to prioritize remote sensing images on CubeSats with more flexibility compared to existing methods. Two approaches implementing the same conceptual prioritization pipeline are compared. The first uses YOLOv8 and Llama2 to extract image features and compare them with text descriptions via cosine similarity. The second approach employs CLIP, fine-tuned on remote sensing data, to achieve the same. Both approaches are evaluated on real nanosatellite hardware, the VERTECS Camera Control Board. The CLIP approach, particularly the ResNet50-based model, shows the best performance in prioritizing and sequencing remote sensing images. This paper demonstrates that on-orbit prioritization using natural language descriptions is viable and allows for more flexibility than existing methods.

![Prioritization Pipeline](https://www.mdpi.com/aerospace/aerospace-11-00888/article_deploy/html/images/aerospace-11-00888-g001.png)

## Oriented Bounding Box (OBB) Approach
Uses fine-tuned YOLOv8 and Llama2 models as image and text feature extractors, respectively.

Two YOLOv8 models with different image input sizes were trained (960 px and 1280 px).

![OBB Pipeline](https://www.mdpi.com/aerospace/aerospace-11-00888/article_deploy/html/images/aerospace-11-00888-g002.png)

## CLIP Approach
Uses CLIP fine-tuned for RS images.

ResNet50 and ViT-B-16 -based CLIP models were trained.

![CLIP Pipeline](https://www.mdpi.com/aerospace/aerospace-11-00888/article_deploy/html/images/aerospace-11-00888-g004.png)

## Datasets
- DOTA-v1.5: https://captain-whu.github.io/DOTA/index.html
- NWPU-Captions: https://ieeexplore.ieee.org/document/9866055

## Citation
Fielding, E.; Hanazawa, A. Flexible Natural Language-Based Image Data Downlink Prioritization for Nanosatellites. Aerospace 2024, 11, 888. https://doi.org/10.3390/aerospace11110888


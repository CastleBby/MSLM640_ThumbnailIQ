# ThumbnailIQ:  
A Computer Vision-Based Framework for Evaluating and Ranking YouTube Thumbnails**

## Overview:  

ThumbnailIQ is a computer vision-based application designed to evaluate and rank YouTube thumbnails based on structural clarity, feature richness, and robustness under transformations.  

Unlike approaches that attempt to reverse-engineer YouTube’s ranking system, this project focuses on measurable visual properties such as edge density, corner detection, keypoints, blur, contrast, and text overlays.  

## How to run:  


## File Organization: 

---
```
thumbnailIQ/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── feature_analysis.ipynb
│
├── src/
│   ├── api/
│   ├── preprocessing/
│   ├── features/
│   ├── robustness/
│   ├── modeling/
│   ├── evaluation/
│
├── outputs/
│   ├── figures/
│   ├── tables/
│
├── app/
│   ├── interface.py
│
├── README.md
└── requirements.txt
```
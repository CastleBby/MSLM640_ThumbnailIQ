# ThumbnailIQ:  
A Computer Vision-Based Framework for Evaluating and Ranking YouTube Thumbnails

## Overview:  

ThumbnailIQ is a computer vision-based application designed to evaluate and rank YouTube thumbnails based on structural clarity, feature richness, and robustness under transformations.  

Unlike approaches that attempt to reverse-engineer YouTube’s ranking system, this project focuses on measurable visual properties such as edge density, corner detection, keypoints, blur, contrast, and text overlays.  

**Clarification** “We observe measurable correlations between certain visual features and engagement metrics, though these relationships do not imply causation.”  

## Broad logic flow:
API ->  URL -> DOWNLOAD -> IMAGE -> CV FEATURES

## How to run:  

**note:** if the order is not specified then API defaults to "relevance"
    this is Youtube's internal ranking most relevant to the search query
    - 


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

## Important Notes:  
- Youtue Data API v3  
10,000 quota units per day for free  
python scripy `youtube_collecto.py` returns 403 error past quota   

- youtube_collector.py 
API script does not collect data on watch time, click through rate, or impressions. 
Therefore engagement is based on views, likes, and comments.  

- Youtube Metrics clarified:  
    - views (in data) = how many times users clicked and watched the video  
    - watch time (not in data) = how long users stayed watching after clicking  
    - impressions (not in data)  = how many times youtube showed thumbnail to users  
    Youtube optimizes for watch time and retention  


# ThumbnailIQ:  
This project evaluates how well keypoint-based feature detection (ORB) preserves meaningful visual structure under real-world distortions, and proposes a robustness-based metric to assess whether an image remains visually effective across viewing conditions.

Key Findings:  


## Overview:  

In practice, thumbnails are displayed across a wide range of environments—small mobile screens, compressed formats, and varying levels of blur. These conditions can degrade the effectiveness of classical computer vision algorithms.

This project investigates:
	- How keypoint detection behaves under real-world distortions
	- How much visual structure is preserved after transformations
	- Whether keypoint-based representations remain reliable across conditions


## Broad logic flow:  
API ->  Thumbnail URL -> Download -> Image -> Keypoint Detection -> Transformations -> Robustness Analysis  

## Reproducibility 
The API calls and collects video meta-data based on the past year from the date ran. Therefore, to reproduce the same results as used in my analysis, the user must input a static date of March 30, 2026. 

**System Dependencies** this project uses OCR features via Tesseract OCT 
install on mac terminal using `brew install tesseract`  

## How to run:  

**note:** if the order is not specified then API defaults to "relevance"
    this is Youtube's internal ranking most relevant to the search query


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
python scripy `youtube_collector.py` returns 403 error past quota   

- youtube_collector.py 
API script does not collect data on watch time, click through rate, or impressions. 
Therefore engagement is based on views, likes, and comments.  

- Youtube Metrics clarified:  
    - views (in data) = how many times users clicked and watched the video  
    - watch time (not in data) = how long users stayed watching after clicking  
    - impressions (not in data)  = how many times youtube showed thumbnail to users  
    Youtube optimizes for watch time and retention  


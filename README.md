# ThumbnailIQ:
ThumbnailIQ is a computer vision application designed to perform visual similarity search on YouTube thumbnails using classical feature detection (ORB keypoints).  


## Overview:  

Given a query thumbnail, the system retrieves visually similar thumbnails from a dataset by matching keypoint features. The project also evaluates how well this matching process performs under real-world distortions such as resizing, compression, blur, and noise.

This simulates a practical scenario where images are viewed across different devices and quality conditions, testing whether classical computer vision methods remain reliable for visual search tasks.

## Problem: 
Content creators often want to understand:
	- What thumbnails look visually similar to theirs
	- What design patterns are common across videos
	- How their thumbnail compares stylistically within a dataset

Modern systems (e.g., reverse image search) rely on deep learning, but this project explores:

Can classical computer vision (keypoint-based matching) be used to perform thumbnail similarity search under real-world conditions?

## Broad logic flow:  

YouTube API -> Thumbnail URL -> Download -> Image -> Keypoint Detection -> Matching -> Similarity Ranking 

Given an input thumbnail, ThumbnailIQ:
	1.	Extracts keypoints using ORB
	2.	Compares the query image to a dataset of thumbnails
	3.	Computes similarity using keypoint matching
	4.	Returns the most visually similar thumbnails

Additionally, the system applies real-world transformations to evaluate:
	- Whether similar images remain detectable under distortion
	- When and why matching begins to fail


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
├── src/
│   ├── data_collection/
│   │   └── youtube_collector.py
│
│   ├── preprocessing/
│   │   └── download_thumbnails.py
│
│   ├── keypoint_analysis/     
│   │   ├── extract_keypoints.py
│   │   ├── transformations.py
│   │   ├── robustness_test.py
│   │   └── retention_score.py
│
│   ├── visualization/
│   │   └── visualize_keypoints.py
│
├── demo/
│   └── run_demo.py             
│
├── notebooks/
│   └── analysis.ipynb     
│
├── outputs/
│   ├── figures/
│   └── results/
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


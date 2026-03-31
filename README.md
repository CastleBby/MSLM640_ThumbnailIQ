# ThumbnailIQ:
ThumbnailIQ is a computer vision application designed to perform visual similarity search on YouTube thumbnails using classical feature detection.  


## Overview:  

Given a query thumbnail, the system retrieves visually similar thumbnails from a dataset using classical computer vision techniques.

This project investigates how different feature representations impact similarity search performance by comparing three approaches:
	•	ORB (baseline) — keypoint detection using FAST + BRIEF descriptors
	•	ORB Hybrid — ORB combined with edge and color similarity
	•	SIFT Hybrid — SIFT combined with edge and color similarity

Rather than relying on deep learning, this project evaluates how far classical methods can go in approximating visual similarity.

## Problem: 
Content creators often want to understand:
	- What thumbnails look visually similar to theirs
	- What design patterns are common across videos
	- How their thumbnail compares stylistically within a dataset

Modern systems (e.g., reverse image search) rely on deep learning, but this project explores:

Can classical computer vision (keypoint-based matching) be used to perform thumbnail similarity search under real-world conditions?

## How to run the demo:  

0. All in one script that bypasses steps 1 - 6 
- user must input path to their image they are testing in "data/raw/images/..."

`bash scripts/run_pipeline.sh data/raw/images/_fXpf-qBca4.jpg`

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

2. install dependencies
`pip install -r requirements.txt`

3. data collection 
```bash
python src/data_collection/youtube_collector.py
```

4. download images

```bash
python src/preprocessing/download_thumbnails.py
```

5. optional feature extraction
```bash
python src/features/extract_features.py
```

6. run the application 
run similarity search 
**user inputs their image here**
```bash
python demo/run_similarity.py data/raw/images/<image_name>.jpg
```

## Broad logic flow:  

YouTube API -> Thumbnail URL -> Download -> Image -> Keypoint Detection -> Matching -> Similarity Ranking 

1. ORB (Baseline)
	•	Uses only keypoint matching
	•	Fast but produces sparse and unstable matches

2. ORB + Edge + Color (Hybrid)
	•	Combines structural, layout, and color features
	•	Improves similarity ranking by incorporating global visual cues

3. SIFT + Edge + Color (Hybrid)
	•	Replaces ORB with SIFT descriptors
	•	Produces more stable and robust matches, especially under transformations

## Findings:
	- ORB alone fails to produce meaningful similarity rankings on real-world thumbnails
	- Hybrid methods significantly improve performance by incorporating layout and color information
	- SIFT-based hybrid methods outperform ORB-based methods, particularly under rotation and structural variation
	- Despite improvements, all classical methods struggle to capture semantic similarity, often matching images with similar textures or layouts but different meanings

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
|   |   └── images/ 
│   ├── processed/
|       └── features.csv
|
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
│   │   └── transformations.py
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
|
├── scripts/
|   └── run_pipeline.sh 
|
├── project_proposals
|   ├── original.md
|   └── updated.md 
|
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


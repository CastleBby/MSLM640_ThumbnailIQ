# Spring 2026 Midterm Project Proposal: ThumbnailIQ — Classical CV for Thumbnail Similarity Search
**Author:** Emily Castelan  
**Date:** March 27, 2026  
**Team Number:** 13  
**Course:** MSLM 640  

---

## Application Description: 

YouTube emphasizes the importance of thumbnails for engagement, yet provides little transparency into how visual properties influence content visibility. While tools such as A/B testing and automatic thumbnail generation exist, they do not offer insight into how computer vision systems interpret and compare thumbnail images.

This project explores a different, more interpretable problem:

Can classical computer vision methods be used to perform visual similarity search on YouTube thumbnails?

Rather than attempting to predict engagement or reverse-engineer YouTube’s ranking system, this application focuses on how classical feature-based methods (edges, corners, and keypoints) behave when comparing real-world images.

The final system, ThumbnailIQ, takes a query thumbnail and retrieves visually similar thumbnails from a dataset using classical CV techniques. The project also evaluates how different feature representations affect similarity performance.

I had originally read background material on my original idea and proposal which referenced: 
"As motivation, Koh and Cui from 2022, explores the relation between the visual attributes of thumbnails and the view-through of videos. The paper looks at features relevant to the aesthetics of thumbnails such as colorfulness, brightness, and image quality. The overall goal of the paper was to provide practical guidelines for designing templates for optimal thumbnails that grab potential viewers' attention and yield more video views. 

Unlike the paper by Koh and Cui, this application does not claim to yield more views, but rather to ensure that the thumbnail chosen by the user meets strong classical computer vision criteria for structural clarity, feature detectability, and robustness under real-world image noise and transformations."

## Data Source:  
The dataset will be built from public YouTube video metadata collected through the YouTube Data API. For each sampled video, the project will store the video ID, channel name, publish date, thumbnail URL, and selected engagement metadata such as views, likes, and comment counts when available through the API. The thumbnail images themselves will be downloaded from the returned thumbnail URLs and used as the input to the CV pipeline. Using the API is appropriate here because it provides structured access to thumbnail and metadata fields without violating YouTube’s prohibition on scraping YouTube applications directly. 

The dataset is constructed using the YouTube Data API.
- Query-based collection using topics such as:
- “study with me”
- “coding setup”
- “notebook aesthetic”
- “underconsumption core”
- “spring trend predictions”
	For each video:
- video ID
- thumbnail URL
- metadata (views, etc. when available)
- Thumbnails are downloaded and used as input to the CV pipeline

This approach ensures:
	•	real-world, unfiltered image data
	•	diverse visual conditions (lighting, clutter, composition)
	•	compliance with API usage policies

## Method Overview

This project evaluates visual similarity between YouTube thumbnails using classical computer vision techniques. Three methods are compared to understand how feature representation impacts performance:

### 1. ORB (Baseline)
- Uses ORB (Oriented FAST and Rotated BRIEF) for keypoint detection and description  
- Fast and efficient, but produces sparse and unstable matches in complex, real-world images  

### 2. ORB Hybrid
- Combines ORB with additional visual features:
  - Edge detection (Canny) for layout information  
  - Color histograms for visual style  
- Improves similarity ranking by incorporating both local structure and global image properties  

### 3. SIFT Hybrid
- Replaces ORB with SIFT (Scale-Invariant Feature Transform) descriptors  
- Combined with edge and color features  
- Produces more stable and robust matches, particularly under transformations such as rotation  

### Final Similarity Score

The similarity between two images is computed as a weighted combination of:

- Keypoint similarity (ORB or SIFT) → structural information  
- Edge similarity → layout and object boundaries  
- Color similarity → visual style and lighting  

This hybrid approach allows the system to approximate visual similarity using only classical CV methods.

## Pipeline Overview

The system follows a modular pipeline from data collection to similarity ranking:

### 1. Data Collection
- Query the YouTube Data API using predefined search terms  
- Retrieve video metadata and thumbnail URLs  

### 2. Image Acquisition
- Download thumbnail images from URLs  
- Store images locally for processing  

### 3. Preprocessing
- Resize images to a fixed resolution for consistency  
- Convert to grayscale for keypoint detection  

### 4. Feature Extraction
- Compute:
  - Keypoints and descriptors (ORB or SIFT)  
  - Edge maps using Canny edge detection  
  - Color histograms for global image representation  

### 5. Similarity Computation
- Compare a query image to all images in the dataset  
- Compute similarity scores using:
  - ORB  
  - ORB + edge + color  
  - SIFT + edge + color  

### 6. Ranking
- Sort images based on similarity score  
- Return the top-K most similar thumbnails  

### 7. Evaluation
- Compare performance across methods  
- Analyze:
  - Visual quality of matches  
  - Stability under real-world variation  
  - Robustness to transformations (e.g., rotation)
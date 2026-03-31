# Spring 2026 Midterm Project Proposal: Evaluating YouTube Thumbnails Using CV Properties 
**Author:** Emily Castelan  
**Date:** March 16, 2026  
**Team Number:** 13  
**Course:** MSLM 640  

---

### Application Description: 

YouTube has vague descriptions which explain that thumbnail selection is important for user engagement, but does not provide explicit guidelines on how the algorithm scores and prioritizes video content based on the thumbnail. Furthermore, while YouTube utilizes thumbnails to order content results, the platform only offers two forms of assistance for thumbnails. The first is an A/B testing option for thumbnails and titles based on watch-time outcomes. The second is automated thumbnail generation from video frames is the user does not select one. While these two options may be helpful, they do not explain to the user the visual attributes that a computer algorithm evaluates. 

This project proposes a thumbnail analysis application that uses classical computer vision to evaluate multiple candidate thumbnails and rank them according to their CV-based robustness and structural clarity. This app does not claim to reverse-engineer YouTube's internal ranking system. Instead, it will use visual properties known to be processed by computer algorithms such as edge structure, corner richness, SIFT keypoints, blur, contract, and test-overlay clutter, to test how stable those properties remain after resizing, compression, rotation, and partial occlusion. 

As motivation, Koh and Cui from 2022, explores the relation between the visual attributes of thumbnails adn the view-through of videos. The paper looks at features relevant to the aesthetics of thumbnails such as colorfulness, brightness, and image quality. The overall goal of the paper was to provide practical guidelines for designing templates for optimal thumbnails that grab potential viewers' attention and yield more video views. 

Unlike the paper by Koh and Cui, this application does not claim to yield more views, but rather to ensure that the thumbnail chosen by the user meets strong classical computer vision criteria for structural clarity, feature detectability, and robustness under real-world image noise and transformations.

### Data Source:  
The dataset will be built from public YouTube video metadata collected through the YouTube Data API. For each sampled video, the project will store the video ID, channel name, publish date, thumbnail URL, and selected engagement metadata such as views, likes, and comment counts when available through the API. The thumbnail images themselves will be downloaded from the returned thumbnail URLs and used as the input to the CV pipeline. Using the API is appropriate here because it provides structured access to thumbnail and metadata fields without violating YouTube’s prohibition on scraping YouTube applications directly. 


### Pipeline Overview:  
 1. Use YouTube API to collect a dataset of videos and their metadata
 2. Download the thumbnail image
 3. Run classical CV pipeline on each thumbnail 
 4. Build a feature table:
 - video_id
 - channel
 - thumbnail_path
 - edge_density
 - corner_count
 - sift_keypoints
 - blur_score
 - contract
 - OCR for text overlays
 - robustness score after resize 
 - popularity & engagement
5. Train a lightweight model on that table
6. Have user input potential thumbnail images 
7. Evaluate potential thumbnails based on CV pipeline 
8. Rank the candidate thumbnails based on the learned model and/or a CV robustness score, and explain the strongest and weakest choices in plain language.
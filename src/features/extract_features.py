""" 
src/features/extract_features.py

Usage: 
------------
This script performs feature extraction on YouTube thumbnail images using
classical computer vision techniques. It transforms image data into a structured
feature dataset that can be used for analysis, comparison, and modeling.


Data Collected:
---------------
	1.	Structural Features:
	- Edge density (Canny edge detection)
	- Corner count (Shi-Tomasi detection)
	- Keypoint count (ORB feature detector)
	2.	Image Quality:
	- Blur score (variance of Laplacian)
	- Contrast (pixel intensity standard deviation)
	3.	Color & Intensity:
	- Colorfulness (RGB channel variation)
	- Brightness (mean grayscale intensity)
	4.	Texture:
	- Entropy (information content of pixel distribution)
	5.	Text Features:
	- Text density (edge-based proxy)
	- Text length (OCR-based)
	- Word count (OCR-based)
	- Text presence (binary indicator)
	6.	Semantic Features:
	- Face count (Haar cascade detection)
	7.	Composition:
	- Symmetry (left-right structural similarity)

Methodology:
------------
	1.	Load metadata and image paths from CSV
	2.	Iterate through each image
	3.	Load image using OpenCV
	4.	Convert image to grayscale where required
	5.	Apply feature extraction functions across multiple domains
	6.	Aggregate extracted features into a structured dataset
	7.	Save results as a CSV file for downstream analysis

Inputs:
------------
data/raw/metadata_with_paths.csv


Outputs:
-------
data/processed/features.csv


"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
import pytesseract


# -----------------------------
# Paths
# -----------------------------
INPUT_CSV = "data/raw/metadata_with_paths.csv"
OUTPUT_CSV = "data/processed/features.csv"

os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Feature Functions
# -----------------------------

# edge detection 
def compute_edge_density(image):
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges > 0) / edges.size

# blur 
def compute_blur(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# corners
def compute_corners(image):
    corners = cv2.goodFeaturesToTrack(image, 100, 0.01, 10)
    return 0 if corners is None else len(corners)

# contrast 
def compute_contrast(image):
    return image.std()

# SIFT keypoints
def compute_keypoints(image):
    orb = cv2.ORB_create(nfeatures=500)
    keypoints = orb.detect(image, None)
    return len(keypoints)

# color 
def compute_colorfulness(image):
    (B, G, R) = cv2.split(image)
    rg = np.abs(R - G)
    yb = np.abs(0.5*(R + G) - B)

    return np.sqrt(rg.std()**2 + yb.std()**2)

# brightness
def compute_brightness(image):
    return np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

# entropy
def compute_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    hist = hist / hist.sum()
    return entropy(hist.flatten())

# text 
""""
def compute_text_density(image):
    edges = cv2.Canny(image, 100, 200)
    return np.mean(edges
"""
def extract_text(image):
    return pytesseract.image_to_string(image)

# face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces(image):
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    return len(faces)

# symmetry 1/3s rule 
def compute_symmetry(image):
    """
    Measures horizontal symmetry of the image.
    Returns a value between 0 and 1 (higher = more symmetric).
    """

    h, w = image.shape[:2]

    # Split image into left and right halves
    left = image[:, :w // 2]
    right = image[:, w // 2:]

    # Flip right half horizontally
    right_flipped = cv2.flip(right, 1)

    # Resize in case of odd width mismatch
    min_width = min(left.shape[1], right_flipped.shape[1])
    left = left[:, :min_width]
    right_flipped = right_flipped[:, :min_width]

    # Compute difference
    diff = np.abs(left.astype("float") - right_flipped.astype("float"))

    # Normalize
    symmetry_score = 1 - (np.mean(diff) / 255)

    return symmetry_score
# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    features = []

    print("Extracting features...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["image_path"]

        try:
            img = cv2.imread(path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # core features:
            edge_density = compute_edge_density(gray)
            blur = compute_blur(gray)
            corners = compute_corners(gray)
            contrast = compute_contrast(gray)

            # additional features 
            keypoints = compute_keypoints(gray)
            colorfulness = compute_colorfulness(img)
            brightness = compute_brightness(img)
            entropy_val = compute_entropy(gray)
            # text_density = compute_text_density(gray)
            text = extract_text(img)
            text_length = len(text.strip())
            word_count = len(text.split())
            text_presence = 1 if len(text.strip()) > 0 else 0
            face_count = detect_faces(gray)
            symmetry = compute_symmetry(gray)

            features.append({
                "video_id": row["video_id"],
                "edge_density": edge_density,
                "blur": blur,
                "corner_count": corners,
                "contrast": contrast,
                "views": row["views"],
                "keypoint_count": keypoints,
                "colorfulness": colorfulness,
                "brightness": brightness,
                "entropy": entropy_val,
                # "text_density": text_density,
                "text": text,
                "text_length": text_length,
                "text_count": word_count,
                "text_presence":text_presence
                "face_count": face_count,
                "symmetry": symmetry,
            })

        except Exception as e:
            print(f"Error processing {path}")
            continue

    feature_df = pd.DataFrame(features)

    feature_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved features to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
""" 
src/robustness/test_robustness.py

Usage: 
------------
This script evaluates the robustness of extracted computer vision features
under common image transformations. It assesses how stable each feature is
when thumbnails are subjected to realistic distortions such as resizing,
compression, blur, and noise.

Data Collected:
---------------
For each image and transformation:
	- Original feature values
	- Transformed feature values
	- Absolute differences between original and transformed features


Methodology:
------------
	1.	Load dataset containing image paths
	2.	For each image:
    a. Extract baseline features using the original image
    b. Apply a set of transformations:
	- Resizing to multiple realistic thumbnail sizes
	- JPEG compression
	- Gaussian blur
	- Additive noise
    c. Extract features from each transformed image
    d. Compute absolute differences between original and transformed features
	3.	Aggregate results across all images
	4.	Compute mean differences per feature and transformation
	5.	Analyze feature sensitivity and robustness

Inputs:
------------
data/raw/metadata_with_paths.csv

Outputs:
-------
"data/processed/robustness_results.csv"
Printed summary statistics of feature differences by transformation


"""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath("."))
from src.features.extract_features import (
    compute_edge_density,
    compute_blur,
    compute_corners,
    compute_contrast,
    compute_keypoints,
    compute_colorfulness,
    compute_brightness,
    compute_entropy,
    detect_faces,
    compute_symmetry
)

INPUT_CSV = "data/raw/metadata_with_paths.csv"

# -----------------------------
# Transformations
# -----------------------------


def resize_variants(img):
    sizes = [
        # Desktop
        (640, 360),
        (480, 270),
        (320, 180),

        # Mobile
        (240, 135),
        (160, 90),

        # Tablet
        (426, 240)
    ]

    resized = {}

    for (w, h) in sizes:
        key = f"resize_{w}x{h}"
        resized[key] = cv2.resize(img, (w, h))

    return resized

def compress_image(img, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def blur_image(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def add_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


# -----------------------------
# variant versions draft 
# -----------------------------

# compression 
# simulates youtube recompression and network degradation 
def compress_variants(img):
    qualities = [90, 60, 30]  # high → medium → low

    out = {}
    for q in qualities:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        out[f"compress_q{q}"] = cv2.imdecode(encimg, 1)

    return out


# blur 
# simulates scaling, smoothing, low quality rendering 
def blur_variants(img):
    kernels = [3, 5, 9]

    out = {}
    for k in kernels:
        out[f"blur_k{k}"] = cv2.GaussianBlur(img, (k, k), 0)

    return out

# noise 
# simulates compression artifacts and re-encoding noise 
def noise_variants(img):
    sigmas = [5, 15, 30]

    out = {}
    for s in sigmas:
        noise = np.random.normal(0, s, img.shape).astype(np.uint8)
        noisy = cv2.add(img, noise)
        out[f"noise_s{s}"] = noisy

    return out




# -----------------------------
# Feature wrapper 
# -----------------------------

def extract_basic_features(img, video_id=None, views=None):
    """
    Extracts full feature set from a single image.
    Returns a dictionary consistent with features.csv structure.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- Core features ----
    edge_density = compute_edge_density(gray)
    blur = compute_blur(gray)
    corners = compute_corners(gray)
    contrast = compute_contrast(gray)

    # ---- Advanced CV features ----
    keypoints = compute_keypoints(gray)
    colorfulness = compute_colorfulness(img)
    brightness = compute_brightness(img)
    entropy_val = compute_entropy(gray)

    # ---- OCR (compute ONCE) ----
    try:
        text = pytesseract.image_to_string(img)
    except:
        text = ""

    cleaned_text = text.strip()

    text_length = len(cleaned_text)
    word_count = len(cleaned_text.split())
    text_presence = 1 if text_length > 0 else 0

    # ---- Semantic ----
    face_count = detect_faces(gray)

    # ---- Composition ----
    symmetry = compute_symmetry(gray)

    # ---- Build output ----
    feature_dict = {
        "video_id": video_id,

        # Core
        "edge_density": edge_density,
        "blur": blur,
        "corner_count": corners,
        "contrast": contrast,

        # Advanced
        "keypoint_count": keypoints,
        "colorfulness": colorfulness,
        "brightness": brightness,
        "entropy": entropy_val,

        # OCR
        "text": text,
        "text_length": text_length,
        "text_count": word_count,
        "text_presence": text_presence,

        # Semantic
        "face_count": face_count,

        # Composition
        "symmetry": symmetry,

        # Target (optional in robustness)
        "views": views
    }

    return feature_dict

# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["image_path"]

        img = cv2.imread(path)
        if img is None:
            continue

        original = extract_basic_features(img)

        NUMERIC_FEATURES = [
            "edge_density",
            "blur",
            "corner_count",
            "contrast",
            "keypoint_count",
            "colorfulness",
            "brightness",
            "entropy",
            "text_length",
            "text_count",
            "text_presence",
            "face_count",
            "symmetry"
        ]
        """
        transforms = {
            **resize_variants(img),
            "compress": compress_image(img),
            "blur": blur_image(img),
            "noise": add_noise(img),
        }
        """
        transforms = {
            **resize_variants(img),
            **compress_variants(img),
            **blur_variants(img),
            **noise_variants(img),
        }

        for name, transformed_img in transforms.items():
            transformed = extract_basic_features(transformed_img)

            for key in NUMERIC_FEATURES :
                if original[key] is None or transformed[key] is None:
                    print(f"skipping {key} due to None value")
                    continue
                diff = abs(original[key] - transformed[key])

                results.append({
                    "video_id": row["video_id"],
                    "feature": key,
                    "transformation": name,
                    "difference": diff
                })

    result_df = pd.DataFrame(results)
    # Save raw results
    os.makedirs("data/processed", exist_ok=True)
    result_df.to_csv("data/processed/robustness_results.csv", index=False)


    print(result_df.groupby(["feature", "transformation"])["difference"].mean())

if __name__ == "__main__":
    main()
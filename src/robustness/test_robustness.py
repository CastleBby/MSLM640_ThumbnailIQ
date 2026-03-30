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

Inputs:
------------
data/raw/metadata_with_paths.csv

Outputs:
-------
Printed summary statistics of feature differences by transformation


"""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

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

        transforms = {
            **resize_variants(img),
            "compress": compress_image(img),
            "blur": blur_image(img),
            "noise": add_noise(img),
        }

        for name, transformed_img in transforms.items():
            transformed = extract_basic_features(transformed_img)

            for key in original:
                diff = abs(original[key] - transformed[key])

                results.append({
                    "feature": key,
                    "transformation": name,
                    "difference": diff
                })

    result_df = pd.DataFrame(results)

    print(result_df.groupby(["feature", "transformation"])["difference"].mean())

if __name__ == "__main__":
    main()
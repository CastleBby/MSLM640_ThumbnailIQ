""" 
src/robustness/test_robustness.py

Usage: 
------------


Data Collected:
---------------


Methodology:
------------

Inputs:
------------


Outputs:
-------


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

def extract_basic_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        "edge_density": compute_edge_density(gray),
        "blur": compute_blur(gray),
        "contrast": compute_contrast(gray),
        "colorfulness": compute_colorfulness(img),
    }

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
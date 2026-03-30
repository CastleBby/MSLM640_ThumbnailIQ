""" 
root/src/preprocessing/download_thumbnails.py
Usage: 
------------
This script downloads YouTube thumbnail images from URLs collected in the
metadata dataset and stores them locally for computer vision processing.


Data Collected:
---------------
image_path: Local file path to each downloaded thumbnail image


Methodology:
------------
	1.	Load metadata CSV containing thumbnail URLs
	2.	Iterate through each row of the dataset
	3.	Download the thumbnail image using HTTP requests
	4.	Save each image using the video_id as the filename
	5.	Avoid duplicate downloads by checking if the file already exists
	6.	Append the local image path to the dataset
	7.	Remove entries where image download failed
	8.	Save the updated dataset with image paths

Inputs:
------------
data/raw/metadata.csv


Outputs:
-------
data/raw/images/
data/raw/metadata_with_paths.csv

"""
import os
import requests
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
INPUT_CSV = "data/raw/metadata.csv"
OUTPUT_CSV = "data/raw/metadata_with_paths.csv"
IMAGE_DIR = "data/raw/images"

os.makedirs(IMAGE_DIR, exist_ok=True)

# -----------------------------
# Download Function
# -----------------------------
def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return True
    except Exception as e:
        print(f"Failed: {url}")
        return False

# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    image_paths = []

    print("Downloading thumbnails...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row["video_id"]
        url = row["thumbnail_url"]

        filename = f"{video_id}.jpg"
        save_path = os.path.join(IMAGE_DIR, filename)

        # Avoid duplicate downloads
        if not os.path.exists(save_path):
            success = download_image(url, save_path)
        else:
            success = True

        if success:
            image_paths.append(save_path)
        else:
            image_paths.append(None)

    df["image_path"] = image_paths

    # Drop failed downloads
    df = df.dropna(subset=["image_path"])

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved updated dataset to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
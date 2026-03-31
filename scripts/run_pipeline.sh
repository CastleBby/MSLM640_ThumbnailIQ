#!/bin/bash

# Check if user provided an argument
if [ -z "$1" ]; then
  echo "Usage: bash scripts/run_pipeline.sh <image_path>"
  exit 1
fi

IMAGE_PATH=$1

echo "Running full pipeline..."

# Step 1: Collect data
python src/data_collection/youtube_collector.py

# Step 2: Download thumbnails
python src/preprocessing/download_thumbnails.py

# Step 3: Run similarity search with user image
python demo/run_similarity.py $IMAGE_PATH
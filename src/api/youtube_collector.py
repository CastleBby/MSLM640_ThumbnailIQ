""" 
root/src/api/youtube_collector.py

Usage: 
------------
This script collects YouTube video metadata using the YouTube Data API v3.
It retrieves a dataset of videos based on a user-defined search query and
stores relevant metadata for downstream computer vision analysis.

Sampling: High-view videos within the past year across multiple content types

Data Collected:
---------------
- video_id: Unique identifier for each video
- channel: Channel name
- publish_date: Video upload timestamp
- title: Video title
- thumbnail_url: URL to the video thumbnail image
- views: Total view count
- likes: Number of likes (if available)
- comments: Number of comments (if available)

Methodology:
------------
1. Use the search().list endpoint to retrieve video IDs based on a query
2. Use the videos().list endpoint to fetch detailed metadata in batches
3. Aggregate results into a structured DataFrame
4. Save the dataset to a CSV file for downstream processing

Output:
-------
data/raw/metadata.csv

"""

import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build
from _datetime import datetime, timedelta

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)

# -----------------------------
# Parameters 
# -----------------------------
SEARCH_QUERIES = [
    "study with me",
    "coding set up",
    "notebook aesthetic",
    "underconsumption core",
    "spring trend predictions"
]
MAX_RESULTS = 550   # for testing changed from 200 -> 500 resulted in 1631 unique 
BATCH_SIZE = 50     # max allowed per API call
# only pull data from videos within the past year
publishedAfter = (datetime.utcnow() - timedelta(days=365)).isoformat("T") + "Z"

# -----------------------------
# Helper: Search Videos
# -----------------------------
def search_videos(query, max_results, order_type):
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            order=order_type, # enfore popularity
            publishedAfter=publishedAfter,
            maxResults=BATCH_SIZE,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response["items"]:
            video_ids.append(item["id"]["videoId"])
        # break # test to stop after first API call 

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return video_ids[:max_results]

# -----------------------------
# Helper: Get Video Metadata
# -----------------------------
def get_video_details(video_ids):
    records = []

    for i in tqdm(range(0, len(video_ids), 50)):
        batch_ids = video_ids[i:i+50]

        request = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch_ids)
        )
        response = request.execute()

        for item in response["items"]:
            snippet = item["snippet"]
            stats = item.get("statistics", {})

            record = {
                "video_id": item["id"],
                "channel": snippet.get("channelTitle"),
                "publish_date": snippet.get("publishedAt"),
                "title": snippet.get("title"),
                "thumbnail_url": snippet["thumbnails"]["high"]["url"],
                "views": stats.get("viewCount"),
                "likes": stats.get("likeCount"),
                "comments": stats.get("commentCount")
            }

            records.append(record)

    return pd.DataFrame(records)

# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    print("Collecting videos across queries...")
    all_video_ids = set()

    per_query = MAX_RESULTS // len(SEARCH_QUERIES)
    per_type = per_query // 2

    for query in SEARCH_QUERIES:
        print(f"Query: {query}")
        #ids = search_videos(query, MAX_RESULTS // len(SEARCH_QUERIES))
        high_ids = search_videos(query, MAX_RESULTS // 2, order_type="viewCount")
        recent_ids = search_videos(query, MAX_RESULTS // 2, order_type="date")

        all_video_ids.update(high_ids)
        all_video_ids.update(recent_ids)

    all_video_ids = list(all_video_ids)

    print(f"Total unique videos collected: {len(all_video_ids)}")

    print("Fetching metadata...")
    df = get_video_details(all_video_ids)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/metadata.csv", index=False)

    print("Saved to data/raw/metadata.csv")


if __name__ == "__main__":
    main()
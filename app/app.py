import streamlit as st
import cv2
import os
import numpy as np

IMAGE_DIR = "data/raw/images"
TOP_K = 5


# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img):
    return cv2.resize(img, (320, 180))


# -----------------------------
# EDGE SIMILARITY
# -----------------------------
def edge_similarity(img1, img2):
    e1 = cv2.Canny(preprocess(img1), 100, 200)
    e2 = cv2.Canny(preprocess(img2), 100, 200)
    return np.sum(e1 == e2) / e1.size


# -----------------------------
# COLOR SIMILARITY
# -----------------------------
def color_similarity(img1, img2):
    img1 = preprocess(img1)
    img2 = preprocess(img2)

    hist1 = cv2.calcHist([img1], [0,1,2], None, [8,8,8], [0,256]*3)
    hist2 = cv2.calcHist([img2], [0,1,2], None, [8,8,8], [0,256]*3)

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# -----------------------------
# SIFT SIMILARITY
# -----------------------------
def sift_similarity(img1, img2):
    sift = cv2.SIFT_create()

    img1 = preprocess(img1)
    img2 = preprocess(img2)

    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return len(good)


# -----------------------------
# FINAL SIMILARITY
# -----------------------------
def similarity(img1, img2):
    return (
        0.4 * sift_similarity(img1, img2) +
        0.3 * edge_similarity(img1, img2) +
        0.3 * color_similarity(img1, img2)
    )


# -----------------------------
# LOAD DATASET
# -----------------------------
def load_images():
    images = {}
    for file in os.listdir(IMAGE_DIR):
        path = os.path.join(IMAGE_DIR, file)
        img = cv2.imread(path)
        if img is not None:
            images[file] = img
    return images


# -----------------------------
# UI
# -----------------------------
st.title("ThumbnailIQ")
st.write("Upload a thumbnail to find visually similar images.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    query_img = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB), caption="Query Image")

    images = load_images()

    scores = []

    for name, img in images.items():
        score = similarity(query_img, img)
        scores.append((name, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    st.subheader("Top Matches")

    cols = st.columns(TOP_K)

    for i in range(TOP_K):
        name, score = scores[i]
        img = images[name]

        with cols[i]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            st.caption(f"Score: {score:.2f}")
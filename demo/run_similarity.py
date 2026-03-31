import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# -----------------------------
# CONFIG
# -----------------------------
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
# FINAL SIMILARITY (SIFT HYBRID)
# -----------------------------
def similarity(img1, img2):
    sift = sift_similarity(img1, img2)
    edge = edge_similarity(img1, img2)
    color = color_similarity(img1, img2)

    return (0.4 * sift) + (0.3 * edge) + (0.3 * color)


# -----------------------------
# LOAD IMAGES
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
# MAIN
# -----------------------------
def main():
    images = load_images()

    # pick query (first image)
    # query_name = list(images.keys())[0]
    # query_img = images[query_name]

    if len(sys.argv) > 1:
        query_path = sys.argv[1]
        query_img = cv2.imread(query_path)

        if query_img is None:
            raise ValueError("Invalid image path provided.")

        query_name = os.path.basename(query_path)
    else:
        query_name = list(images.keys())[0]
        query_img = images[query_name]

    scores = []

    for name, img in images.items():
        if name == query_name:
            continue

        score = similarity(query_img, img)
        scores.append((name, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # -----------------------------
    # VISUALIZE RESULTS
    # -----------------------------
    fig, ax = plt.subplots(1, TOP_K + 1, figsize=(15,5))

    # query
    ax[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Query")
    ax[0].axis("off")

    # top matches
    for i, (name, score) in enumerate(scores[:TOP_K]):
        img = images[name]
        ax[i+1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[i+1].set_title(f"{score:.2f}")
        ax[i+1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
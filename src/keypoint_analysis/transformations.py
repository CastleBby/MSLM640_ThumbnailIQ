import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.keypoint_analysis.extract_keypoints import extract_keypoints, draw_keypoints


# -----------------------------
# Transformations
# -----------------------------
def resize_variants(img):
    sizes = [
        (640, 360),
        (480, 270),
        (320, 180),
        (240, 135),
        (160, 90),
        (426, 240)
    ]

    out = {}
    for (w, h) in sizes:
        out[f"resize_{w}x{h}"] = cv2.resize(img, (w, h))

    return out


def compress_variants(img):
    qualities = [90, 60, 30]

    out = {}
    for q in qualities:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        out[f"compress_q{q}"] = cv2.imdecode(encimg, 1)

    return out


def blur_variants(img):
    kernels = [3, 5, 9]

    out = {}
    for k in kernels:
        out[f"blur_k{k}"] = cv2.GaussianBlur(img, (k, k), 0)

    return out


def noise_variants(img):
    sigmas = [5, 15, 30]

    out = {}
    for s in sigmas:
        noise = np.random.normal(0, s, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        out[f"noise_s{s}"] = noisy

    return out


# -----------------------------
# Main Logic
# -----------------------------
def run_analysis(image_path):
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Failed to load image")

    original_kp, original_count = extract_keypoints(img)

    print(f"\nOriginal Keypoints: {original_count}")

    transforms = {
        **resize_variants(img),
        **compress_variants(img),
        **blur_variants(img),
        **noise_variants(img),
    }

    results = {}

    for name, t_img in transforms.items():
        kp, count = extract_keypoints(t_img)

        retention = count / original_count if original_count > 0 else 0

        results[name] = {
            "count": count,
            "retention": retention
        }

    print("\n--- Keypoint Retention Results ---")
    for k, v in results.items():
        print(f"{k}: count={v['count']}, retention={v['retention']:.2f}")

    return img, original_kp, transforms, results


# -----------------------------
# Visualization
# -----------------------------
def visualize(img, original_kp, transforms, name):
    if name not in transforms:
        print("Transform not found")
        return

    t_img = transforms[name]

    kp_t, count_t = extract_keypoints(t_img)

    img_kp = draw_keypoints(img, original_kp)
    img_kp_t = draw_keypoints(t_img, kp_t)

    img_kp = cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB)
    img_kp_t = cv2.cvtColor(img_kp_t, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img_kp)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(img_kp_t)
    ax[1].set_title(name)
    ax[1].axis("off")

    plt.show()


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    IMAGE_PATH = "data/raw/images/_fXpf-qBca4.jpg"

    img, original_kp, transforms, results = run_analysis(IMAGE_PATH)

    visualize(img, original_kp, transforms, "resize_160x90")
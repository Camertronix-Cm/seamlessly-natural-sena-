import os
import sys
import csv
import cv2
import numpy as np
import urllib.request
from urllib.error import HTTPError, URLError

# =============================================
# 🔧 BRISQUE (no-reference IQA)
#=============================================

DATASET_DIR = "our results"

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------
def ensure_quality_module():
    if not hasattr(cv2, "quality"):
        raise RuntimeError(
            "cv2.quality not found.\n"
            "Install opencv-contrib-python:\n"
            "pip uninstall -y opencv-python opencv-python-headless\n"
            "pip install opencv-contrib-python"
        )

# ------------------------------------------------------------
# Download helpers
# ------------------------------------------------------------
def download_first_available(urls, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if os.path.exists(dst_path):
        return

    last_err = None
    for url in urls:
        try:
            print(f"Downloading: {os.path.basename(dst_path)}")
            urllib.request.urlretrieve(url, dst_path)
            return
        except (HTTPError, URLError) as e:
            last_err = e

    raise RuntimeError(
        f"Failed to download {os.path.basename(dst_path)}\n"
        f"Last error: {last_err}"
    )

def ensure_brisque_models(models_dir):
    model_path = os.path.join(models_dir, "brisque_model_live.yml")
    range_path = os.path.join(models_dir, "brisque_range_live.yml")

    branches = ["4.x", "main", "master"]
    base_urls = [
        f"https://raw.githubusercontent.com/opencv/opencv_contrib/{b}/modules/quality/samples/"
        for b in branches
    ]

    model_urls = [u + "brisque_model_live.yml" for u in base_urls]
    range_urls = [u + "brisque_range_live.yml" for u in base_urls]

    download_first_available(model_urls, model_path)
    download_first_available(range_urls, range_path)

    return model_path, range_path

# ------------------------------------------------------------
# BRISQUE metric
# ------------------------------------------------------------
def compute_brisque(img_bgr, brisque_obj):
    return float(brisque_obj.compute(img_bgr)[0])

# ------------------------------------------------------------
# Dataset traversal
# ------------------------------------------------------------
def list_images(dataset_dir):
    paths = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXT):
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths

def safe_mean(vals):
    return float(np.mean(vals)) if vals else float("nan")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    ensure_quality_module()

    if not os.path.isdir(DATASET_DIR):
        raise ValueError(f"Folder not found: {DATASET_DIR}")

    image_paths = list_images(DATASET_DIR)
    if not image_paths:
        raise ValueError(f"No images in: {DATASET_DIR}")

    print(f"Analyzing {len(image_paths)} images...\n")

    # BRISQUE setup
    models_dir = os.path.join(DATASET_DIR, "models")
    brisque_model, brisque_range = ensure_brisque_models(models_dir)
    brisque_obj = cv2.quality.QualityBRISQUE_create(
        brisque_model, brisque_range
    )

    brisque_scores = []
    rows = []
    ok = 0

    for i, path in enumerate(image_paths, start=1):
        rel = os.path.relpath(path, DATASET_DIR)
        print(f"[{i}/{len(image_paths)}] {rel}")

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            rows.append([rel, "", "FAIL_LOAD"])
            continue

        try:
            b = compute_brisque(img, brisque_obj)

            brisque_scores.append(b)
            ok += 1

            rows.append([rel, f"{b:.6f}", "OK"])

        except Exception as e:
            rows.append([rel, "", f"FAIL:{type(e).__name__}"])
            print(f"   ❌ Error: {e}")

    mean_b = safe_mean(brisque_scores)

    print("\n" + "=" * 50)
    print(f"BRISQUE RESULTS")
    print(f"Images found     = {len(image_paths)}")
    print(f"Images evaluated = {ok}")
    print(f"Mean BRISQUE (↓ better) = {mean_b:.3f}")
    print("=" * 50)

    # Save results
    out_txt = os.path.join(DATASET_DIR, "brisque_summary.txt")
    out_csv = os.path.join(DATASET_DIR, "brisque_details.csv")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {os.path.abspath(DATASET_DIR)}\n")
        f.write(f"Images: {len(image_paths)}\n")
        f.write(f"Evaluated: {ok}\n\n")
        f.write(f"Mean BRISQUE: {mean_b:.6f}\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "brisque", "status"])
        w.writerows(rows)

    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
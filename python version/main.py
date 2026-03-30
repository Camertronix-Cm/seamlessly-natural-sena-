
import torch
import numpy as np
import imageio as imio
import os
import cv2
import tqdm
import matplotlib.pyplot as plt
import random
import statistics
import copy
import math
import heapq
import time
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import multiprocessing
import threading
from scipy.ndimage import gaussian_filter
from curveConcatenation import CurveConcatenationLine
from warpImages import warp_images_with_xfeat_points
from researchAdequateZone import research_with_real_timeout


xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)


new_curveconcatenationline = CurveConcatenationLine()


def crop_black_borders(image, threshold=10):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image[0:0, 0:0], (0, 0, 0, 0)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]
    return cropped


def image_stitching(im1, im2, pair_index=0):

    debut = time.time()

    im1_undistorted = im1
    im2_undistorted = im2
    
    print("Matches extraction ...")
    mkpts_0, mkpts_1 = xfeat.match_xfeat(im1_undistorted, im2_undistorted, top_k=4096, min_cossim=-1)

    print("Warping of images ...")
    im1_undistorted, im2_undistorted, mkpts_0, mkpts_1, Aglob, ox, oy, (Hs, Ws), (Ht, Wt) = warp_images_with_xfeat_points(
        im1_undistorted, im2_undistorted, mkpts_0, mkpts_1
    )

  
    canvas_img1 = im1_undistorted.copy()
    canvas_img2 = im2_undistorted.copy()

    # ===== GRAYSCALE =====
    if len(canvas_img1.shape) == 3:
        gray1 = cv2.cvtColor(canvas_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(canvas_img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = canvas_img1
        gray2 = canvas_img2

    # ===== MASKS =====
    _, maskA = cv2.threshold(gray1, 10, 255, cv2.THRESH_BINARY)
    _, maskB = cv2.threshold(gray2, 10, 255, cv2.THRESH_BINARY)

    maskA = maskA.astype(bool)
    maskB = maskB.astype(bool)

    mask_onlyA = maskA & (~maskB)
    mask_onlyB = maskB & (~maskA)
    mask_overlap = maskA & maskB

    # =====================================================
    # OVERLAP ANALYSIS
    # =====================================================

    area_A = np.sum(maskA)
    area_B = np.sum(maskB)
    area_overlap = np.sum(mask_overlap)
    area_union = np.sum(maskA | maskB)

    overlap_ratio = area_overlap / area_union if area_union > 0 else 0
    overlap_A = area_overlap / area_A if area_A > 0 else 0
    overlap_B = area_overlap / area_B if area_B > 0 else 0

    # =====================================================
    # POLYGONS
    # =====================================================

    src_poly = np.float32([
        [0, 0],
        [Ws-1, 0],
        [Ws-1, Hs-1],
        [0, Hs-1]
    ]).reshape(-1,1,2)

    src_poly_warped = cv2.perspectiveTransform(src_poly, Aglob).reshape(-1,2)
    polyA = src_poly_warped + np.array([[ox, oy]], dtype=np.float32)

    polyB = np.float32([
        [0, 0],
        [Wt-1, 0],
        [Wt-1, Ht-1],
        [0, Ht-1]
    ]) + np.array([[ox, oy]], dtype=np.float32)

    def point_in_polygon(pt, polygon):
        poly = polygon.astype(np.float32)
        pt = (float(pt[0]), float(pt[1]))
        return cv2.pointPolygonTest(poly, pt, False) >= 0

    A_in_B = [point_in_polygon(pt, polyB) for pt in polyA]
    num_A_in_B = sum(A_in_B)

    B_in_A = [point_in_polygon(pt, polyA) for pt in polyB]
    num_B_in_A = sum(B_in_A)

    # =====================================================
    # EARLY EXIT CONDITION
    # =====================================================

    if overlap_A >= 0.98 or overlap_B >= 0.98 or num_A_in_B == 4 or num_B_in_A == 4:

        if num_A_in_B == 4:
         
            result = canvas_img2

        elif num_B_in_A == 4:
            result = canvas_img1

        else:
            result = canvas_img1 if area_A >= area_B else canvas_img2

        result = crop_black_borders(result)

        return result

    transformed = research_with_real_timeout(mkpts_0, mkpts_1, timeout=5)

    if transformed is None or len(transformed) == 0:
        transformed = [[p1, p2] for p1, p2 in zip(mkpts_0, mkpts_1)]

    filtered_list_transformed = []

    current_list = transformed[0]
    filtered_list_transformed.append(current_list)

    for next_list in transformed[1:]:

        current_y = current_list[1][1]
        next_y = next_list[1][1]

        if next_y > current_y:
            filtered_list_transformed.append(next_list)
            current_list = next_list

    transformed = filtered_list_transformed

    matched_kp1 = [pair[0] for pair in transformed]
    matched_kp2 = [pair[1] for pair in transformed]

    intensities1, _, _, _, _ = new_curveconcatenationline.get_scharr_intensities(im1, matched_kp1)
    intensities2, _, _, _, _ = new_curveconcatenationline.get_scharr_intensities(im2, matched_kp2)

    differences = new_curveconcatenationline.compute_intensity_differences(intensities1, intensities2)

    matched_kp1, matched_kp2 = new_curveconcatenationline.filter_and_match(differences)

    matched_kp1, matched_kp2 = new_curveconcatenationline.remove_duplicates(matched_kp1, matched_kp2)
    matched_kp2, matched_kp1 = new_curveconcatenationline.remove_duplicates(matched_kp2, matched_kp1)

    label_matched_kp1 = new_curveconcatenationline.labelling_points(matched_kp1)
    label_matched_kp2 = new_curveconcatenationline.labelling_points(matched_kp2)

    path_ids, path_points = new_curveconcatenationline.find_path(label_matched_kp1)

    points_img2_img1 = new_curveconcatenationline.getting_points_img2_corresponding(label_matched_kp2, path_ids)

    im3 = im1_undistorted.copy()
    im4 = im2_undistorted.copy()

    print("Selection process of the stitching line's keypoints ...")

    vertical_points1, mask1, path_points1, check1 = new_curveconcatenationline.plot_lines_on_image(im1_undistorted, points=path_points, case="im1")
    vertical_points2, mask2, path_points2, check2 = new_curveconcatenationline.plot_lines_on_image(im2_undistorted, points=points_img2_img1, case="im2")

    print("Refining the selection process of the stitching line's keypoints ...")

    path_points1, path_points2 = new_curveconcatenationline.filter_lists(path_points1, path_points2)

    path_points1, path_points2 = new_curveconcatenationline.validate_and_filter_points_by_ratio(path_points1, path_points2, min_ratio=0.7, max_ratio=1)

    print("Horizontal slicing ...")

    segments1 = new_curveconcatenationline.horizontal_cut(im3, path_points1)
    segments2 = new_curveconcatenationline.horizontal_cut(im4, path_points2)

    list_dimensions1, list_points_adapt1 = new_curveconcatenationline.list_dimensions_list_points_adapt(segments1, path_points1)
    list_dimensions2, list_points_adapt2 = new_curveconcatenationline.list_dimensions_list_points_adapt(segments2, path_points2)

    print("Horizontal assembling of segments ...")

    result_horizontals, shift_first, _, list_cut = new_curveconcatenationline.process_segments(
        segments1, segments2, list_points_adapt1, list_points_adapt2
    )

    result_horiz_images = [t[0] for t in result_horizontals if t and t[0] is not None]

    result_vertical = new_curveconcatenationline.process_segmentsV(result_horiz_images, matched_kp1)

    
    result_vertical = crop_black_borders(result_vertical)

    return result_vertical


def get_image_files(directory):
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith(('.jpg', '.png', '.JPG', '.PNG', '.jpeg'))]


if __name__ == "__main__":
  
    
    dataset_dir = 'datasets'
    results_dir = 'results1'

    os.makedirs(results_dir, exist_ok=True)

    image_files = get_image_files(dataset_dir)


    pairs_done = 0
    total_time = 0.0

    for i in range(0, len(image_files), 2):
        if i + 1 >= len(image_files):
            break

        im1 = cv2.imread(image_files[i])
        im2 = cv2.imread(image_files[i + 1])

        if im1 is None or im2 is None:
            print(f"❌ Lecture échouée pour: {image_files[i]} ou {image_files[i+1]}")
            continue

        start_time = time.time()
        print(f"🔧 Image processing: {image_files[i]}  &  {image_files[i + 1]}")

        try:
            result_vertical = image_stitching(im1, im2)
        except Exception as e:
            print(f"❌ Erreur dans image_stitching: {e}")
            continue

        # Save results
        result_filename = f"stitched_{i//2 + 1}.jpg"
        cv2.imwrite(os.path.join(results_dir, result_filename), result_vertical)
        print(f"💾 Saved image : {result_filename}")

        elapsed = time.time() - start_time
        total_time += elapsed
        pairs_done += 1
        print(f"⏱️ Running time for this pair: {elapsed:.4f} seconds\n")

    # ======= Average times =======

    if pairs_done > 0:
        print(f"Total time              : {total_time:.2f} s")
        print(f"Avg time per pair       : {total_time / pairs_done:.2f} s")
    print("=============================")
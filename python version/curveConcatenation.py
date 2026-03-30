
import torch
import numpy as np
import os
import cv2
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


class CurveConcatenationLine:
    def __init__(self):
        """Initialization"""
        pass

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def find_nearest_below(self, current, points):
        _, (x1, y1) = current
        candidates = [p for p in points if p[1][1] > y1]
        if not candidates:
            return None
        return min(candidates, key=lambda p: self.distance((x1, y1), p[1]))

    def find_path(self, points):
        path_ids = []
        path_points = []
        points = sorted(points, key=lambda x: x[1])
        current = points[0]
        path_ids.append(current[0])
        path_points.append(current[1])
        points.remove(current)

        while points:
            next_point = self.find_nearest_below(current, points)
            if next_point is None:
                break
            path_ids.append(next_point[0])
            path_points.append(next_point[1])
            points.remove(next_point)
            current = next_point

        return path_ids, path_points

    def draw_points_on_image(self, points, image_path1, image_path):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for x, y in points:
            if 0 <= x < image.width and 0 <= y < image.height:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
            else:
                print(f"Point ({x}, {y}) is out of bounds.")
        image.show()
        image.save(image_path1)

    def labelling_points(self, points):
        return [[i + 1, point] for i, point in enumerate(points)]

    def tracer_lignes_et_points_cv(self, image, points, nom_fichier='/content/sample_data/test.jpg'):
        image = image.copy()
        if image is None:
            print("Error: Unable to load image.")
            return
        hauteur, largeur, _ = image.shape
        centre_x = largeur // 2
        centre_y = hauteur // 2
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
        cv2.line(image, (centre_x, 0), (centre_x, hauteur), (0, 255, 0), 2)
        cv2.line(image, (0, centre_y), (largeur, centre_y), (0, 255, 0), 2)

    def getting_points_img2_corresponding(self, label_matched_kp2, path_ids):
        tuples_recuperes = []
        for pos in path_ids:
            if 1 <= pos <= len(label_matched_kp2):
                tuples_recuperes.append(label_matched_kp2[pos - 1][1])
        return tuples_recuperes

    def find_closest_points(self, points1, points2):
        closest_top_left = min(points1, key=lambda p: (p[0], p[1]))
        corresponding_top_left = points2[points1.index(closest_top_left)]
        closest_bottom_right = max(points1, key=lambda p: (p[0], p[1]))
        corresponding_bottom_right = points2[points1.index(closest_bottom_right)]
        closest_left = sorted(points1, key=lambda p: p[0])
        corresponding_left = [points2[points1.index(p)] for p in closest_left]
        closest_right = sorted(points1, key=lambda p: p[0], reverse=True)
        corresponding_right = [points2[points1.index(p)] for p in closest_right]
        closest_bottom = sorted(points1, key=lambda p: p[1], reverse=True)
        corresponding_bottom = [points2[points1.index(p)] for p in closest_bottom]
        latest_point = points1[-1]
        latest_point_corresponding = points2[-1]
        return (latest_point, latest_point_corresponding, closest_left, corresponding_left,
                closest_top_left, corresponding_top_left, closest_right[-1], corresponding_right,
                closest_bottom[-1], corresponding_bottom, closest_bottom_right, corresponding_bottom_right)

    def New_translation(self, image1_path, image2_path, closest_left, corresponding_left, mask_path, rx):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
        image2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)
        if image1.shape[0] != image2.shape[0]:
            raise ValueError("The two images must be the same height.")
        if image1.shape[2] == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2BGRA)
        if image2.shape[2] == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2BGRA)
        height, width1, _ = image1.shape
        _, width2, _ = image2.shape
        if rx > 0:
            black_band = np.zeros((height, rx, 4), dtype=np.uint8)
            adjusted_image1 = np.hstack((image1, black_band))
            adjusted_image2 = np.hstack((black_band, image2))
            padding_right = np.zeros((height, rx), dtype=mask.dtype)
            mask = np.hstack((mask, padding_right))
        elif rx < 0:
            rx = abs(rx)
            black_band = np.zeros((height, rx, 4), dtype=np.uint8)
            adjusted_image1 = np.hstack((black_band, image1))
            adjusted_image2 = np.hstack((image2, black_band))
            padding_left = np.zeros((height, rx), dtype=mask.dtype)
            mask = np.hstack((padding_left, mask))
        else:
            adjusted_image1 = image1
            adjusted_image2 = image2
        mask_normalized = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        if adjusted_image1.shape[:2] != adjusted_image2.shape[:2] or adjusted_image1.shape[:2] != mask_normalized.shape:
            raise ValueError("The dimensions of the images and the mask do not match after adjustment.")
        result = adjusted_image1.copy()
        result[mask_normalized == 0] = adjusted_image2[mask_normalized == 0]
        return result

    def plot_lines_on_image(self, img, points, case):
        """Extracting points to form the stitching line"""
        if img is None:
            raise ValueError("The image could not be loaded. Check the image path.")

        height, width, _ = img.shape
        all_points = []
        mask = np.ones((height, width), dtype=np.uint8) * 255

        first_point = min(points, key=lambda p: p[1])
        last_point = max(points, key=lambda p: p[1])
        
        start_point = (int(round(first_point[0])), 0)
        end_point = (int(round(first_point[0])), int(round(first_point[1])))
        
        # ✅ DEFINITION DE check AVANT UTILISATION
        check = self.get_vertical_trajectory_points(start_point, int(round(first_point[1])))
        all_points.extend(check)

        for i in range(0, len(points) - 1, 2):
            a = (int(round(points[i][0])), int(round(points[i][1])))
            b = (int(round(points[i + 1][0])), int(round(points[i + 1][1])))
            c = (int((a[0] + b[0]) // 2), int(min(a[1], b[1])))

            all_points.extend(self.get_hypotenuse_points(a, c))
            all_points.extend(self.get_hypotenuse_points(b, c))

            cv2.line(img, a, c, (0, 255, 255), 2)
            cv2.line(img, b, c, (0, 255, 255), 2)
            
            if i + 2 < len(points):
                a_prime = (int(round(points[i + 2][0])), int(round(points[i + 2][1])))
                cv2.line(img, b, a_prime, (0, 255, 255))
        
        vertical_end_1080 = (int(round(last_point[0])), height)
        all_points.extend(self.get_vertical_trajectory_points(
            (int(round(last_point[0])), int(round(last_point[1]))), height))

        for j in range(len(all_points) - 1):
            cv2.line(img, all_points[j], all_points[j + 1], (0, 255, 255), 2)

        if case == "im1":
            for y in range(height):
                row = img[y, :, :]
                yellow_pixel_index = np.where((row[:, 0] == 0) & (row[:, 1] == 255) & (row[:, 2] == 255))[0]
                if yellow_pixel_index.size > 0:
                    first_yellow_index = yellow_pixel_index[0]
                    if first_yellow_index < width:
                        row[first_yellow_index:] = [0, 0, 0]
                        mask[y, first_yellow_index:] = 0
                        
        if case == "im2":
            for y in range(height):
                row = img[y, :, :]
                yellow_pixel_index = np.where((row[:, 0] == 0) & (row[:, 1] == 255) & (row[:, 2] == 255))[0]
                if yellow_pixel_index.size > 0:
                    first_yellow_index = yellow_pixel_index[0]
                    if first_yellow_index > 0:
                        row[:first_yellow_index] = [0, 0, 0]
                        mask[y, :first_yellow_index] = 0

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # ✅ RETOUR AVEC check BIEN DÉFINI
        return all_points, mask, points, check
    def get_vertical_trajectory_points(self, start_point, end_y):
        x = start_point[0]
        trajectory_points = []
        for y in range(start_point[1], end_y + 1):
            trajectory_points.append((x, y))
        return trajectory_points

    def get_hypotenuse_points(self, point1, point2):
        points = []
        x1, y1 = point1
        x2, y2 = point2
        dist = int(np.linalg.norm(np.array(point2) - np.array(point1)))
        if dist == 0:
            return [point1]
        for i in range(dist + 1):
            x = int(round(x1 + (x2 - x1) * i / dist))
            y = int(round(y1 + (y2 - y1) * i / dist))
            points.append((x, y))
        return points

    def horizontal_cut(self, img, liste1):
 
        segments = []
        img_height, img_width = img.shape[:2]
        
        for i in range(len(liste1)):
            upper = 0 if i == 0 else int(liste1[i - 1][1])
            lower = int(liste1[i][1])
            
            if upper < lower <= img_height:
                # Cut image segment
                segment = img[upper:lower, 0:img_width]
                segments.append(segment)           
              
            else:
                print(f" Invalid data for slicing at index {i}: upper={upper}, lower={lower}, img_height={img_height}")
        
        # Add remaining bottom segment if exists
        if liste1 and liste1[-1][1] < img_height:
            lower = int(liste1[-1][1])
            segment = img[lower:img_height, 0:img_width]
            segments.append(segment)    
       
        return segments

    def validate_and_filter_points_by_ratio(self, liste1, liste2, min_ratio=0.7, max_ratio=1.3):
        if len(liste1) != len(liste2):
            raise ValueError("The list must have the same length.")
        l1 = list(liste1)
        l2 = list(liste2)
        i = 1
        while i < len(l1):
            try:
                y_prev1 = float(l1[i-1][1])
                y_curr1 = float(l1[i][1])
                y_prev2 = float(l2[i-1][1])
                y_curr2 = float(l2[i][1])
            except (ValueError, TypeError, IndexError) as e:
                raise ValueError(f"Invalid data from {i-1} ou {i}: {e}")
            h1 = y_curr1 - y_prev1
            h2 = y_curr2 - y_prev2
            if h1 <= 0 or h2 <= 0:
                print(f"[Height is null or negative -> Point suppressed{i}")
                del l1[i]
                del l2[i]
                continue
            ratio = h2 / h1
            if not (min_ratio <= ratio <= max_ratio):
                del l1[i]
                del l2[i]
            else:
                i += 1
        return l1, l2

    def list_dimensions_list_points_adapt(self, segments1, liste1):
        list_dimensions = []
        list_points_adapt = []
        for i, segment in enumerate(segments1):
            if isinstance(segment, np.ndarray):
                height, width = segment.shape[:2]
                list_dimensions.append((width, height))
            else:
                raise ValueError(f"The segment at index {i} is not a valid image object.")
        if liste1:
            first_width, first_height = liste1[0]
            list_points_adapt.append((first_width, 0))
            list_points_adapt.append((first_width, first_height))
            for i in range(len(liste1) - 1):
                current_width, current_height = liste1[i]
                next_width, next_height = liste1[i + 1]
                list_points_adapt.append((current_width, 0))
                list_points_adapt.append((next_width, abs(next_height - current_height)))
            last_height = list_dimensions[-1][1]
            last_width = liste1[-1][0]
            list_points_adapt.append((last_width, 0))
            list_points_adapt.append((last_width, last_height))
        return list_dimensions, list_points_adapt

    def add_black_band(self, image, height_diff, position='top'):
        if height_diff <= 0:
            return image
        band = np.zeros((int(height_diff), image.shape[1], 3), dtype=np.uint8)
        if position == 'top':
            new_image = np.vstack([band, image])
        elif position == 'bottom':
            new_image = np.vstack([image, band])
        return new_image

    def adjust_intermediate_height(self, image, original_width, original_height, target_height):
        if image.shape[0] == target_height:
            return image
        aspect_ratio = original_width / original_height
        new_width = int(target_height * aspect_ratio)
        resized_image = cv2.resize(image, (original_width, target_height), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def update_coordinates(self, original_coords, height_diff):
        x, y = original_coords
        new_coords = (x, y + abs(height_diff))
        return new_coords

    def create_mask(self, image_shape, pt1, pt2, position):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        height, width = image_shape[:2]
        if position == 'left':
            points = np.array([[0, 0], [int(round(pt1[0])), int(round(pt1[1]))], [int(round(pt2[0])), int(round(pt2[1]))], [0, height]])
        elif position == 'right':
            points = np.array([[width, 0], [int(round(pt1[0])), int(round(pt1[1]))], [int(round(pt2[0])), int(round(pt2[1]))], [width, height]])
        elif position == 'middle':
            points = np.array([[int(round(pt1[1])), int(round(pt1[0]))], [int(round(pt2[1])), int(round(pt2[0]))], [int(round(pt2[1])), height], [int(round(pt1[0])), height]])
        else:
            raise ValueError("Invalid position specified. Choose from 'left', 'right', or 'middle'.")
        cv2.fillPoly(mask, [points], 255)
        return mask

    def blend_middle_regions(self, cp1_middle, cp2_middle):
        min_height = min(cp1_middle.shape[0], cp2_middle.shape[0])
        min_width = min(cp1_middle.shape[1], cp2_middle.shape[1])
        cp1_middle_resized = cv2.resize(cp1_middle, (min_width, min_height))
        cp2_middle_resized = cv2.resize(cp2_middle, (min_width, min_height))
        cp1_middle_float = cp1_middle_resized.astype(np.float32)
        cp2_middle_float = cp2_middle_resized.astype(np.float32)
        height, width, _ = cp1_middle_resized.shape
        alpha_cp1 = np.linspace(1, 0, width).reshape(1, width, 1).repeat(height, axis=0)
        alpha_cp2 = 1 - alpha_cp1
        blended_middle = (cp1_middle_float * alpha_cp1 + cp2_middle_float * alpha_cp2).astype(np.uint8)
        return blended_middle

    def sub_concatenation_middle(self, img1, img2, points1, points2):
        if img1 is None or img2 is None:
            raise ValueError("One or both input images are None")
        mask_cp1 = self.create_mask(img1.shape, points1[0], points1[1], 'left')
        mask_cp2 = self.create_mask(img2.shape, points2[0], points2[1], 'middle')
        mask_cp3 = self.create_mask(img2.shape, points2[0], points2[1], 'right')
        cp1 = cv2.bitwise_and(img1, img1, mask=mask_cp1)
        cp2 = cv2.bitwise_and(img2, img2, mask=mask_cp2)
        cp3 = cv2.bitwise_and(img2, img2, mask=mask_cp3)
        if points1[0][0] < points1[1][0]:
            cp1 = cp1[:, :int(round(points1[0][0]))]
            cp1_middle = img1[:, int(round(points1[0][0])):int(round(points1[1][0]))]
            cp2_middle = img2[:, int(round(points2[0][0])):int(round(points2[1][0]))]
            cp3 = cp3[:, int(round(points2[1][0])):]
        if points1[0][0] >= points1[1][0]:
            cp1 = cv2.bitwise_and(img1, img1, mask=mask_cp1)
            cp1 = img1[:, :int(round(points1[1][0]))]
            cp1_middle = img1[:, int(round(points1[1][0])):int(round(points1[0][0]))]
            cp2_middle = img2[:, int(round(points2[1][0])):int(round(points2[0][0]))]
            cp3 = img2[:, int(round(points2[0][0])):]
        blended_middle = self.blend_middle_regions(cp1_middle, cp2_middle)
        result = np.hstack((cp1, blended_middle, cp3))
        return result, cp1_middle, cp2_middle, blended_middle


    def sub_concatenation_first_last(self, img1, img2, points1, points2):
        """
        Concatenation for FIRST and LAST segments
        Transition de ~50px entre img1 et img2, puis 100% img2
        """
       

        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images")

        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        # ✅ TRANSITION FIXE DE 50 PIXELS
        transition_width = 50
        
        # Point de couture (coordonnée X du point de stitching)
        stitch_x1 = int(round(points1[0][0]))
        stitch_x2 = int(round(points2[0][0]))

        # Zone de transition : 50px AVANT et APRÈS le point de couture
        x1_start = max(0, stitch_x1 - transition_width // 2)
        x1_end = min(width1, stitch_x1 + transition_width // 2)
        x2_start = max(0, stitch_x2 - transition_width // 2)
        x2_end = min(width2, stitch_x2 + transition_width // 2)

        # ✅ S'assurer qu'on a bien 50px de transition
        if x1_end - x1_start < transition_width:
            # Élargir si nécessaire
            x1_start = max(0, x1_end - transition_width)
        if x2_end - x2_start < transition_width:
            x2_start = max(0, x2_end - transition_width)

        # --- DÉCOUPAGE ---
        # Partie GAUCHE : img1 pure (avant transition)
        cp1 = img1[:, :x1_start]
   
        # Zone de TRANSITION : blend img1 ↔ img2 (50px)
        cp1_middle = img1[:, x1_start:x1_end]
     

        cp2_middle = img2[:, x2_start:x2_end]
        
        
        # Partie DROITE : img2 PURE (après transition) ✅ CRITIQUE
        cp3 = img2[:, x2_end:]
     
        # ✅ GESTION DES CAS VIDES - Surtout pour cp3 !
        if cp1.size == 0 or cp1.shape[1] == 0:
            # Pas de problème, on commence directement par la transition
            cp1 = np.zeros((height1, 0, 3), dtype=np.uint8) if len(img1.shape) == 3 else np.zeros((height1, 0), dtype=np.uint8)
        
        if cp3.size == 0 or cp3.shape[1] == 0:
            # ⚠️ CRITIQUE : Si cp3 est vide, on prend les derniers pixels de img2
            # AU LIEU de créer du noir !
            cp3 = img2[:, max(0, width2 - 50):width2].copy()
            print(f"  ⚠️ cp3 vide → extension des derniers pixels de img2")

        # Ajustement hauteur
        min_height = min(cp1_middle.shape[0], cp2_middle.shape[0])
        
        if cp1_middle.shape[0] != min_height and cp1_middle.size > 0:
            cp1_middle = cv2.resize(cp1_middle, (cp1_middle.shape[1], min_height), interpolation=cv2.INTER_LINEAR)
        if cp2_middle.shape[0] != min_height and cp2_middle.size > 0:
            cp2_middle = cv2.resize(cp2_middle, (cp2_middle.shape[1], min_height), interpolation=cv2.INTER_LINEAR)
     

        # ✅ BLENDING INTELLIGENT AVEC MASKS (uniquement sur zone valide)
        blended_middle = self.smart_blend_with_masks(cp1_middle, cp2_middle, transition_width)
        cv2.imwrite("blended_middle.png",blended_middle)
        # Blur léger pour lisser
        #blended_middle = cv2.GaussianBlur(blended_middle, (5, 5), 0)

        # Ajustement latéral
        if cp1.size > 0 and cp1.shape[0] != min_height:
            cp1 = cv2.resize(cp1, (cp1.shape[1], min_height), interpolation=cv2.INTER_LINEAR)
        if cp3.size > 0 and cp3.shape[0] != min_height:
            cp3 = cv2.resize(cp3, (cp3.shape[1], min_height), interpolation=cv2.INTER_LINEAR)

        # --- ASSEMBLAGE FINAL ---
        # [ img1 pure ] + [ transition 50px ] + [ img2 pure ]
        result = np.hstack((cp1, blended_middle, cp3))
        
        return result


    def smart_blend_with_masks(self, imgA, imgB, transition_width=50):
        """
        Blending intelligent :
        - Uniquement là où les DEUX images ont de l'information
        - Dégradé linéaire sur transition_width pixels
        """
        if imgA.shape[:2] != imgB.shape[:2]:
            min_h = min(imgA.shape[0], imgB.shape[0])
            min_w = min(imgA.shape[1], imgB.shape[1])
            imgA = cv2.resize(imgA, (min_w, min_h), interpolation=cv2.INTER_LINEAR)
            imgB = cv2.resize(imgB, (min_w, min_h), interpolation=cv2.INTER_LINEAR)
        
        H, W = imgA.shape[:2]
        
        # Masks de validité (pixels non-noirs)
        if len(imgA.shape) == 3:
            grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
        else:
            grayA = imgA
            grayB = imgB
        
        threshold = 10
        valid_A = grayA > threshold
        valid_B = grayB > threshold
        
        # Zone où les DEUX sont valides → blend
        overlap_valid = valid_A & valid_B
        
        # Zone où SEULEMENT A est valide → prendre A
        only_A = valid_A & (~valid_B)
        
        # Zone où SEULEMENT B est valide → prendre B
        only_B = (~valid_A) & valid_B
        
        result = np.zeros_like(imgA)
        
        # 1. Zone only A
        if np.any(only_A):
            result[only_A] = imgA[only_A]
        
        # 2. Zone only B
        if np.any(only_B):
            result[only_B] = imgB[only_B]
        
        # 3. Zone overlap → dégradé linéaire
        if np.any(overlap_valid):
            # Dégradé de 1→0 de gauche à droite
            alpha = np.linspace(1, 0, W, dtype=np.float32)
            if len(imgA.shape) == 3:
                alpha = alpha.reshape(1, -1, 1)
            else:
                alpha = alpha.reshape(1, -1)
            
            blend_zone = (imgA.astype(np.float32) * alpha + 
                         imgB.astype(np.float32) * (1 - alpha))
            result[overlap_valid] = np.clip(blend_zone[overlap_valid], 0, 255).astype(np.uint8)
        else:
            # Pas de overlap → prendre A ou B
            result = imgA.copy()
            if np.any(valid_B):
                result[valid_B] = imgB[valid_B]
        
        return result
    def blend_middle_regions_simple(self, cp1_middle, cp2_middle):
        h, w1 = cp1_middle.shape[:2]
        _, w2 = cp2_middle.shape[:2]
        target_w = (w1 + w2) // 2
        cp1_res = cv2.resize(cp1_middle, (target_w, h), interpolation=cv2.INTER_LINEAR)
        cp2_res = cv2.resize(cp2_middle, (target_w, h), interpolation=cv2.INTER_LINEAR)
        alpha = np.linspace(1, 0, target_w, dtype=np.float32)
        if len(cp1_res.shape) == 3:
            alpha = alpha.reshape(1, -1, 1)
        else:
            alpha = alpha.reshape(1, -1)
        blended = cp1_res * alpha + cp2_res * (1 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def concatenate_images(self, segment1, segment2, points1, points2, is_first=False, is_last=False):
        try:
            shift_first = 0
            cp1_middle = cp2_middle = blended_middle = None
            concatenated = None
            if is_first:
                height_diff = points1[1][1] - points2[1][1]
                shift_first = height_diff
                if height_diff < 0:
                    segment1 = self.add_black_band(segment1, abs(height_diff), position='top')
                else:
                    segment2 = self.add_black_band(segment2, abs(height_diff), position='top')
                concatenated = self.sub_concatenation_first_last(segment1, segment2, points1, points2)
                cp1_middle = segment1.copy()
                cp2_middle = segment2.copy()
                blended_middle = concatenated.copy() if concatenated is not None else None
            elif is_last:
                height_diff = points2[1][1] - points1[1][1]
                shift_first = height_diff
                if height_diff < 0:
                    segment2 = self.add_black_band(segment2, abs(height_diff), position='bottom')
                else:
                    segment1 = self.add_black_band(segment1, abs(height_diff), position='bottom')
                concatenated = self.sub_concatenation_first_last(segment1, segment2, points1, points2)
                cp1_middle = segment1.copy()
                cp2_middle = segment2.copy()
                blended_middle = concatenated.copy() if concatenated is not None else None
            else:
                height1, height2 = segment1.shape[0], segment2.shape[0]
                width1, width2 = segment1.shape[1], segment2.shape[1]
                target_height = max(height1, height2)
                height_diff1 = points1[1][1] - points2[1][1]
                if height1 < target_height:
                    segment1 = self.adjust_intermediate_height(segment1, width1, height1, target_height)
                    points1[1] = self.update_coordinates(points1[1], height_diff1)
                if height2 < target_height:
                    segment2 = self.adjust_intermediate_height(segment2, width2, height2, target_height)
                    points2[1] = self.update_coordinates(points2[1], height_diff1)
                concatenated, cp1_middle, cp2_middle, blended_middle = self.sub_concatenation_middle(segment1, segment2, points1, points2)
        
            
            return concatenated, shift_first, cp1_middle, cp2_middle, blended_middle

        except Exception as e:
            print(f"Error in concatenating images: {e}")
            return None, None, None, None, None

    def process_segments(self, segments1, segments2, points_list1, points_list2):
        result_horizontals = []
        list_cut = []
        shift_first_global = None
        used = {"image1": [], "image2": []}
        if len(segments1) == 0 or len(segments2) == 0:
            print("No segments to process.")
            return result_horizontals, shift_first_global, used, list_cut
        for i, (segment1, segment2) in enumerate(zip(segments1, segments2)):
            if i * 2 + 1 >= len(points_list1) or i * 2 + 1 >= len(points_list2):
                print(f"Missing points for the segment {i}.")
                result_horizontals.append(None)
                list_cut.append(None)
                continue
            points1 = points_list1[i * 2: (i + 1) * 2]
            points2 = points_list2[i * 2: (i + 1) * 2]
            is_first = (i == 0)
            is_last = (i == len(segments1) - 1)
            cp1_middle = cp2_middle = blended_middle = None
            result = shift_first = None
            pts1_used, pts2_used = points1, points2
            try:
                output = self.concatenate_images(segment1, segment2, points1, points2, is_first, is_last)
                if not isinstance(output, (tuple, list)) or len(output) != 5:
                    raise ValueError(f"Expected 5 return values, got {len(output) if hasattr(output, '__len__') else type(output)}")
                ret, shift_first, cp1_middle, cp2_middle, blended_middle = output
                if ret is None:
                    raise ValueError("Image résultante est None")
                result = ret
                result_horizontals.append((result, pts1_used, pts2_used))
                list_cut.append((cp1_middle, cp2_middle, blended_middle))
                if shift_first_global is None and shift_first is not None:
                    shift_first_global = shift_first
                used["image1"].extend([tuple(map(float, p)) for p in pts1_used])
                used["image2"].extend([tuple(map(float, p)) for p in pts2_used])
            except Exception as e:
                print(f"Error during segment processing {i}: {e}")
                result_horizontals.append(None)
                list_cut.append(None)
        return result_horizontals, shift_first_global, used, list_cut

    
    def resize_images_to_average_width(self, images):
        average_width = int(sum(img.shape[1] for img in images) / len(images))
        resized_images = [
            cv2.resize(img, (average_width, int(round(img.shape[0] * (average_width / img.shape[1])))))
            for img in images
        ]
        return resized_images

    def concatenate_images_vertically(self, images):
        max_width = max(img.shape[1] for img in images)
        resized_images = [cv2.copyMakeBorder(img, 0, 0, 0, max_width - img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for img in images]
        return np.vstack(resized_images)

    def process_images_vertically(self, images, output_dir):
        if not isinstance(images, list) or not all(isinstance(img, np.ndarray) for img in images):
            raise ValueError("The images parameter must be a list of image matrices (OpenCV).")
        resized_images = self.resize_images_to_average_width(images)
        result = self.concatenate_images_vertically(resized_images)
        output_path = os.path.join(output_dir, f'{len(images) + 3}.png')
        cv2.imwrite(output_path, result)
        return result

    def process_segmentsV(self, segments, list1):
        if not segments:
            print("Error: the segments list is empty.")
            self.used_points_vertical = {"image1": [], "image2": []}
            return None, None
        valid_segments = []
        meta = []
        for i, seg in enumerate(segments):
            if isinstance(seg, (list, tuple)) and len(seg) >= 1 and isinstance(seg[0], np.ndarray):
                img = seg[0]
                pts1 = seg[1] if len(seg) > 1 and seg[1] is not None else []
                pts2 = seg[2] if len(seg) > 2 and seg[2] is not None else []
            else:
                img = seg
                pts1, pts2 = [], []
            if self.is_null_image(img):
                print(f"The segments[{i}] image is null or invalid. It will be ignored.")
                continue
            valid_segments.append(img)
            h, w = img.shape[:2]
            def _norm_pts(P):
                out = []
                for p in P:
                    try:
                        x, y = float(p[0]), float(p[1])
                        out.append((x, y))
                    except Exception:
                        pass
                return out
            meta.append((w, h, _norm_pts(pts1), _norm_pts(pts2)))
        if not valid_segments:
            print("Error: no valid image found.")
            self.used_points_vertical = {"image1": [], "image2": []}
            return None, None
        average_width = int(sum(img.shape[1] for img in valid_segments) / len(valid_segments))
        resized_images = []
        y_offsets = []
        y_acc = 0
        used_vertical = {"image1": [], "image2": []}
        for (w, h, pts1, pts2), img in zip(meta, valid_segments):
            if w <= 0:
                scale = 1.0
                new_h = h
                img_r = img
            else:
                scale = float(average_width) / float(w)
                new_h = int(round(h * scale))
                img_r = cv2.resize(img, (average_width, new_h), interpolation=cv2.INTER_LINEAR)
            if pts1:
                used_vertical["image1"].extend([(x * scale, y * scale + y_acc) for (x, y) in pts1])
            if pts2:
                used_vertical["image2"].extend([(x * scale, y * scale + y_acc) for (x, y) in pts2])
            resized_images.append(img_r)
            y_offsets.append(y_acc)
            y_acc += new_h
        result_vertical = np.vstack(resized_images)
        self.used_points_vertical = used_vertical
        return result_vertical

    def is_null_image(self, img):
        if img is None:
            return True
        if not isinstance(img, np.ndarray):
            return True
        if img.size == 0:
            return True
        if img.max() == img.min():
            return True
        return False

    def filter_lists(self, L1, L2):
        filtered_L1 = [L1[0]]
        filtered_L2 = [L2[0]]
        for i in range(len(L1) - 1):
            x1_current, y1_current = filtered_L1[-1]
            x1_next, y1_next = L1[i + 1]
            x2_current, y2_current = filtered_L2[-1]
            x2_next, y2_next = L2[i + 1]
            if (x1_current > x1_next and x2_current > x2_next) or (x1_current < x1_next and x2_current < x2_next):
                filtered_L1.append((x1_next, y1_next))
                filtered_L2.append((x2_next, y2_next))
        return filtered_L1, filtered_L2

    def remove_duplicates(self, L1, L2):
        i = 0
        while i < len(L1) - 1:
            if L1[i][0] == L1[i + 1][0]:
                del L1[i + 1]
                del L2[i + 1]
            else:
                i += 1
        return L1, L2

    def sort_and_filter(self, L1, L2):
        combined = list(zip(L1, L2))
        combined.sort(key=lambda x: x[1][1])
        filtered = []
        seen = set()
        for l1, l2 in combined:
            if l2[1] not in seen:
                filtered.append((l1, l2))
                seen.add(l2[1])
        L1_filtered, L2_filtered = zip(*filtered)
        return L1_filtered, L2_filtered

    def get_scharr_intensities(self, image, points):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image.copy()
        grad_x = cv2.Scharr(image_gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(image_gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        intensities = {}
        directions = {}
        list_features_we_gets_intensity = []
        max_top_intensity = -1.0
        max_bottom_intensity = -1.0
        top_point = None
        bottom_point = None
        def bilinear_interpolation(x, y, img):
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)
            dx, dy = x - x0, y - y0
            I00, I10 = img[y0, x0], img[y0, x1]
            I01, I11 = img[y1, x0], img[y1, x1]
            I_top = (1 - dx) * I00 + dx * I10
            I_bottom = (1 - dx) * I01 + dx * I11
            I_final = (1 - dy) * I_top + dy * I_bottom
            return float(I_final)
        for (x, y) in points:
            if 0 <= x < image_gray.shape[1] and 0 <= y < image_gray.shape[0]:
                intensity = bilinear_interpolation(x, y, magnitude)
                direction = bilinear_interpolation(x, y, angle)
                intensities[(x, y)] = intensity
                directions[(x, y)] = direction
                list_features_we_gets_intensity.append((x, y))
                if top_point is None or (y < top_point[1]) or (y == top_point[1] and intensity > max_top_intensity):
                    max_top_intensity = intensity
                    top_point = (x, y)
                if bottom_point is None or (y > bottom_point[1]) or (y == bottom_point[1] and intensity > max_bottom_intensity):
                    max_bottom_intensity = intensity
                    bottom_point = (x, y)
            else:
                intensities[(x, y)] = None
                directions[(x, y)] = None
        return intensities, list_features_we_gets_intensity, directions, top_point, bottom_point

    def compute_intensity_differences(self, intensities1, intensities2):
        differences = []
        if len(intensities1) != len(intensities2):
            raise ValueError("Both intensity dictionaries must have the same number of points.")
        for ((x1, y1), intensity1), ((x2, y2), intensity2) in zip(intensities1.items(), intensities2.items()):
            intensity1 = intensity1 if intensity1 is not None else 0
            intensity2 = intensity2 if intensity2 is not None else 0
            intensity_diff = intensity1 - intensity2
            differences.append([(x1, y1), (x2, y2), intensity_diff])
        return differences

    def filter_and_match(self, differences):
        if not differences:
            return [], [], None
        intensity_diffs = [diff[2] for diff in differences]
        first_quartile = np.percentile(intensity_diffs, 50)
        filtered_diffs = [(point1, point2, intensity_diff) for point1, point2, intensity_diff in differences if intensity_diff <= first_quartile]
        if filtered_diffs:
            mean_of_quartile = np.mean([diff[2] for diff in filtered_diffs])
        else:
            mean_of_quartile = None
        match1 = [point1 for point1, _, _ in filtered_diffs]
        match2 = [point2 for _, point2, _ in filtered_diffs]
        return match1, match2

    def index_longest_list(self, lists):
        if not lists:
            return None
        longest_index = 0
        longest_length = len(lists[0])
        for index, sublist in enumerate(lists):
            if len(sublist) > longest_length:
                longest_length = len(sublist)
                longest_index = index
        return longest_index
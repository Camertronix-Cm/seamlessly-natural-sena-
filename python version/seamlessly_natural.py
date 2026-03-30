
import torch
import numpy as np
import imageio as imio
import os
import cv2
import matplotlib.pyplot as plt
import random
import statistics
import copy
import math
import heapq
import time
from google.colab.patches import cv2_imshow
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import multiprocessing
import threading
from scipy.ndimage import gaussian_filter

xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)
#Load some example imag

class CurveConcatenationLine:

    def __init__(self):
        """Initialization"""
        #self.img1_path = img1_path
        #self.img2_path = img2_path

    #### BEGINNING OF THE POINT EXTRACTION PROCESS THAT SHOULD BE PART OF THE STITCHING LINE.####

    def distance(self,p1, p2):

        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def find_nearest_below(self,current, points):

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

    def draw_points_on_image(self,points, image_path1, image_path):

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for x, y in points:

            if 0 <= x < image.width and 0 <= y < image.height:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
            else:
                print(f"Point ({x}, {y}) is out of bounds.")


        image.show()
        image.save(image_path1)

    def labelling_points(self,points):

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



    def getting_points_img2_corresponding(self,label_matched_kp2,path_ids):
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
        return latest_point, latest_point_corresponding,closest_left, corresponding_left, closest_top_left, corresponding_top_left, closest_right[-1], corresponding_right,closest_bottom[-1], corresponding_bottom,closest_bottom_right, corresponding_bottom_right


    def New_translation(self,image1_path, image2_path,closest_left, corresponding_left, mask_path,rx):

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
            widthF, heightF=result.shape[:2]

            return result


    def plot_lines_on_image(self,img,points,case):
        ## Extracting points to form the stitching line
        if img is None:
            raise ValueError("The image could not be loaded. Check the image path.")

        height, width, _ = img.shape
        all_points = []

        mask = np.ones((height, width), dtype=np.uint8) * 255

        first_point = min(points, key=lambda p: p[1])

        last_point = max(points, key=lambda p: p[1])
        start_point = (int(round(first_point[0])), 0)
        end_point = (int(round(first_point[0])), int(round(first_point[1])))
        all_points.extend(self.get_vertical_trajectory_points(start_point, int(round(first_point[1]))))
        check =self.get_vertical_trajectory_points(start_point,int(round(first_point[1])))

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
        all_points.extend(self.get_vertical_trajectory_points((int(round(last_point[0])), int(round(last_point[1]))), height))

        for j in range(len(all_points) - 1):
            cv2.line(img, all_points[j], all_points[j + 1], (0, 255, 255), 2)

        if case=="im1":

          for y in range(height):
              row = img[y, :, :]
              yellow_pixel_index = np.where((row[:, 0] == 0) & (row[:, 1] == 255) & (row[:, 2] == 255))[0]

              if yellow_pixel_index.size > 0:

                  first_yellow_index = yellow_pixel_index[0]

                  if first_yellow_index < width:
                      row[first_yellow_index:] = [0, 0, 0]
                      mask[y, first_yellow_index:] = 0
        if case =="im2":

          for y in range(height):
              row = img[y, :, :]
              yellow_pixel_index = np.where((row[:, 0] == 0) & (row[:, 1] == 255) & (row[:, 2] == 255))[0]

              if yellow_pixel_index.size > 0:

                  first_yellow_index = yellow_pixel_index[0]

                  if first_yellow_index > 0:
                      row[:first_yellow_index] = [0, 0, 0]
                      mask[y, :first_yellow_index] = 0

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return all_points,mask,points,check

    #### BEGINNING OF THE TASKS AIMED AT SELECTING, FROM THE POINTS OBTAINED ABOVE,
    ####THOSE THAT ENABLE SAFE TRAVERSAL OF THE IMAGE ALONG THE VERTICAL AXIS WITHIN THE FREE PARALLAX ZONE.

    def get_vertical_trajectory_points(self,start_point, end_y):

        x = start_point[0]
        trajectory_points = []

        for y in range(start_point[1], end_y + 1):
            trajectory_points.append((x, y))

        return trajectory_points

    def get_hypotenuse_points(self,point1, point2):
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

    #### SLICING BOTH IMAGES HORIZONTALLY USING THE COORDINATES OBTAINED ABOVE.


    def horizontal_cut(self,img, liste1):
        segments = []
        img_height, img_width = img.shape[:2]
        for i in range(len(liste1)):

            upper = 0 if i == 0 else int(liste1[i - 1][1])
            lower = int(liste1[i][1])
            if upper < lower <= img_height:
                segment = img[upper:lower, 0:img_width]
                segments.append(segment)
            else:
                print("invalid data for slicing")
        if liste1 and liste1[-1][1] < img_height:
            lower = int(liste1[-1][1])
            segment = img[lower:img_height, 0:img_width]
            segments.append(segment)

        return segments

    def validate_and_filter_points_by_ratio(self, liste1, liste2, min_ratio=0.7, max_ratio=1.3):
        """
        Validate the points by comparing the RATIO of segment heights between the two images.

        Remove inconsistent points (whose h2/h1 ratio is outside of [min_ratio, max_ratio]).
        """
        if len(liste1) != len(liste2):
            raise ValueError("The list must have the same length .")

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
                #print(f"[FILTRE RATIO] Point suppressed {i} : h1={h1:.2f}, h2={h2:.2f}, ratio={ratio:.3f} ∉ [{min_ratio}, {max_ratio}]")

                del l1[i]
                del l2[i]

            else:
                i += 1

        return l1, l2


    def list_dimensions_list_points_adapt(self,segments1, liste1):
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
    # Creation of a black band
    def add_black_band(self,image, height_diff, position='top'):
        if height_diff <= 0:
            return image
        band = np.zeros((int(height_diff), image.shape[1], 3), dtype=np.uint8)
        if position == 'top':
            new_image = np.vstack([band, image])
        elif position == 'bottom':
            new_image = np.vstack([image, band])
        return new_image

    def adjust_intermediate_height(self,image,original_width,original_height,target_height):

        if image.shape[0] == target_height:
            return image

        aspect_ratio = original_width / original_height
        new_width = int(target_height * aspect_ratio)

        resized_image = cv2.resize(image, (original_width, target_height), interpolation=cv2.INTER_LINEAR)

        return resized_image

    def update_coordinates(self,original_coords, height_diff):

        x, y = original_coords
        new_coords = (x, y + abs(height_diff))

        return new_coords

    def create_mask(self,image_shape, pt1, pt2, position):
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

    #### ASSEMBLING AND BLENDING OF ALL SEGMENTS.

    def blend_middle_regions(self,cp1_middle, cp2_middle):
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

    def sub_concatenation_middle(self,img1, img2, points1, points2):

        if img1 is None or img2 is None:
            raise ValueError("One or both input images are None")
        mask_cp1 = self.create_mask(img1.shape, points1[0], points1[1], 'left')
        mask_cp2 = self.create_mask(img2.shape, points2[0], points2[1], 'middle')
        mask_cp3 = self.create_mask(img2.shape, points2[0], points2[1], 'right')
        cp1 = cv2.bitwise_and(img1, img1, mask=mask_cp1)
        cp2 = cv2.bitwise_and(img2, img2, mask=mask_cp2)
        cp3 = cv2.bitwise_and(img2, img2, mask=mask_cp3)

        if points1[0][0]<points1[1][0]:
            cp1 = cp1[:, :int(round(points1[0][0]))]
            cp1_middle = img1[:, int(round(points1[0][0])):int(round(points1[1][0]))]
            cp2_middle = img2[:, int(round(points2[0][0])):int(round(points2[1][0]))]
            cp3 = cp3[:, int(round(points2[1][0])):]

        if points1[0][0]>=points1[1][0]:
            cp1 = cv2.bitwise_and(img1, img1, mask=mask_cp1)
            cp1 = img1[:, :int(round(points1[1][0]))]

            cp1_middle = img1[:, int(round(points1[1][0])):int(round(points1[0][0]))]
            cp2_middle = img2[:, int(round(points2[1][0])):int(round(points2[0][0]))]
            cp3 = img2[:, int(round(points2[0][0])):]
        blended_middle = self.blend_middle_regions(cp1_middle, cp2_middle)


        result = np.hstack((cp1, blended_middle, cp3))

        return result,cp1_middle,cp2_middle, blended_middle

    def sub_concatenation_first_last(self, img1, img2, points1, points2):
        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images")

        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        overlap_width = 10
        x1_start = int(round(points1[0][0] - overlap_width))
        x1_end = int(round(points1[0][0] + overlap_width))
        x2_start = int(round(points2[0][0] - overlap_width))
        x2_end = int(round(points2[0][0] + overlap_width))

        x1_start = max(0, x1_start)
        x1_end = min(width1, x1_end)
        x2_start = max(0, x2_start)
        x2_end = min(width2, x2_end)

        # --- Slicing ---
        cp1 = img1[:, :x1_start]          # left
        cp1_middle = img1[:, x1_start:x1_end]
        cp2_middle = img2[:, x2_start:x2_end]
        cp3 = img2[:, x2_end:]            # right

        # --- Checking if the lateral's segments are not empty ---
        if cp1.size == 0:
            cp1 = np.zeros((height1, 5, 3), dtype=np.uint8) if len(img1.shape) == 3 else np.zeros((height1, 5), dtype=np.uint8)
        if cp3.size == 0:
            cp3 = np.zeros((height2, 5, 3), dtype=np.uint8) if len(img2.shape) == 3 else np.zeros((height2, 5), dtype=np.uint8)

        min_height = min(cp1_middle.shape[0], cp2_middle.shape[0])

        cp1_middle_resized = cv2.resize(cp1_middle, (cp1_middle.shape[1], min_height), interpolation=cv2.INTER_LINEAR)
        cp2_middle_resized = cv2.resize(cp2_middle, (cp2_middle.shape[1], min_height), interpolation=cv2.INTER_LINEAR)

        # ---  BLENDING WITH LIGHT FLUO ---
        blended_middle = self.blend_middle_regions_simple(cp1_middle_resized, cp2_middle_resized)

        # --- LIGHT BLUR ---
        blended_middle = cv2.GaussianBlur(blended_middle, (3, 3), 0)

        # --- RESIZE DIFFERENT SEGMENTS ---
        if cp1.size > 0 and cp1.shape[0] != min_height:
            cp1 = cv2.resize(cp1, (cp1.shape[1], min_height), interpolation=cv2.INTER_LINEAR)
        if cp3.size > 0 and cp3.shape[0] != min_height:
            cp3 = cv2.resize(cp3, (cp3.shape[1], min_height), interpolation=cv2.INTER_LINEAR)

        # --- FINAL ASSEMBLING ---
        result = np.hstack((cp1, blended_middle, cp3))
        return result


    def blend_middle_regions_simple(self, cp1_middle, cp2_middle):

        h, w1 = cp1_middle.shape[:2]
        _, w2 = cp2_middle.shape[:2]

        # RESIZE AT THE SAME WIDTH (AVERAGE)
        target_w = (w1 + w2) // 2
        cp1_res = cv2.resize(cp1_middle, (target_w, h), interpolation=cv2.INTER_LINEAR)
        cp2_res = cv2.resize(cp2_middle, (target_w, h), interpolation=cv2.INTER_LINEAR)

        # LINEAR ALPHA FROM 1 to 0
        alpha = np.linspace(1, 0, target_w, dtype=np.float32)
        if len(cp1_res.shape) == 3:
            alpha = alpha.reshape(1, -1, 1)
        else:
            alpha = alpha.reshape(1, -1)

        blended = cp1_res * alpha + cp2_res * (1 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def concatenate_images(self, segment1, segment2, points1, points2, is_first=False, is_last=False):
        """ Function in charge to assembly all the segments together and get the final result"""
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

                # GENERATION OF SEGMENTS
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

                # GENERATION OF SEGMENTS
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
          """
        Processes pairs of image segments and concatenates them horizontally.
        Returns the concatenated images, the global offset, the points used, and the intermediate cuts.
          """
          result_horizontals = []
          list_cut = []
          shift_first_global = None
          used = {"image1": [], "image2": []}

          # basic Verification
          if len(segments1) == 0 or len(segments2) == 0:
              print(" No segments to process.")
              return result_horizontals, shift_first_global, used, list_cut

          for i, (segment1, segment2) in enumerate(zip(segments1, segments2)):

              # Check that points exist for this segment
              if i * 2 + 1 >= len(points_list1) or i * 2 + 1 >= len(points_list2):
                  print(f" Missing points for the segment {i}.")
                  result_horizontals.append(None)
                  list_cut.append(None)
                  continue

              # Extraction of corresponding points
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

    #### EXTRACTION OF THE OVERLAPPING ZONE FROM BOTH IMAGES

    def extraction_usefull_area(self, image1, image2, points1, points2,
                            keep1="left", keep2="right",
                            color=(0,0,255), thickness=2):


        def _center_crop(img, H, W):
            h, w = img.shape[:2]
            y0 = max((h - H)//2, 0); x0 = max((w - W)//2, 0)
            return img[y0:y0+H, x0:x0+W]

        def _crop_to_common_min(*arrs):
            H = min(a.shape[0] for a in arrs); W = min(a.shape[1] for a in arrs)
            return tuple(_center_crop(a, H, W) for a in arrs)

        def _normalize_points(points, W, H):
            if points is None: return []
            arr = np.asarray(points, dtype=np.float32).reshape(-1,2)
            if arr.size == 0: return []
            arr[:,0] = np.clip(arr[:,0], 0, W-1); arr[:,1] = np.clip(arr[:,1], 0, H-1)
            return [(int(x), int(y)) for x,y in arr]

        def _sanitize_points_monotone_y(pts, eps=1):
            if not pts: return []
            clean, last_y = [], -10**9
            for x,y in pts:
                if y >= last_y + eps:
                    clean.append((x,y)); last_y = y
            return clean

        def _build_poly(points, H):
            if not points: return []
            acc = []
            x1, y1 = points[0]
            acc += [(x1, 0), (x1, y1)]
            for (xa, ya), (xb, yb) in zip(points, points[1:]):
                acc += [(xb, ya), (xb, yb)]
            acc.append((points[-1][0], H - 1))
            dedup = [acc[0]]
            for p in acc[1:]:
                if p != dedup[-1]:
                    dedup.append(p)
            return dedup

        def _draw_and_masks(img, pts_raw):
            H, W = img.shape[:2]
            pts = _sanitize_points_monotone_y(_normalize_points(pts_raw, W, H))

            if not pts:
                out = img.copy()
                line_bin = np.zeros((H, W), np.uint8)
                mleft = np.zeros((H, W), np.uint8)
                mright = np.ones((H, W),  np.uint8)
                return out, line_bin, mleft, mright
            thickness=4
            poly = _build_poly(pts, H)

            out = img.copy()
            cv2.polylines(out, [np.array(poly, np.int32)], False, color, thickness, lineType=cv2.LINE_AA)

            line_img = np.zeros((H, W), np.uint8)
            cv2.polylines(line_img, [np.array(poly, np.int32)], False, 255, 1, lineType=cv2.LINE_AA)

            frontiere_x = np.full(H, -1, dtype=np.int32)
            ys, xs = np.where(line_img > 0)

            if ys.size > 0:
                y_min, y_max = ys.min(), ys.max()
                for y in range(y_min, y_max + 1):
                    xs_at_y = xs[ys == y]
                    if xs_at_y.size > 0:
                        frontiere_x[y] = xs_at_y.max()

                last = -1
                for y in range(y_min, y_max + 1):
                    if frontiere_x[y] >= 0:
                        last = frontiere_x[y]
                    else:
                        frontiere_x[y] = last if last >= 0 else 0
                for y in range(0, y_min):
                    frontiere_x[y] = frontiere_x[y_min]
                for y in range(y_max + 1, H):
                    frontiere_x[y] = frontiere_x[y_max]
            else:

                x0 = int(np.clip(poly[0][0], 0, W - 1))
                frontiere_x[:] = x0

            mleft  = np.zeros((H, W), np.uint8)
            mright = np.zeros((H, W), np.uint8)
            for y in range(H):
                x = int(np.clip(frontiere_x[y], 0, W - 1))
                if x > 0:
                    mleft[y, :x] = 1
                mright[y, x:] = 1

            line_bin = (line_img > 0).astype(np.uint8)
            return out, line_bin, mleft, mright
        img1, img2 = _crop_to_common_min(image1, image2)
        out1, line1, m1_left, m1_right = _draw_and_masks(img1, points1)
        out2, line2, m2_left, m2_right = _draw_and_masks(img2, points2)

        mask_refine1 = m1_left if keep1 == "left" else m1_right
        mask_refine2 = m2_left if keep2 == "left" else m2_right
        res1 = (out1, line1, mask_refine1)
        res2 = (out2, line2, mask_refine2)
        return res1, res2

    #### VERTICAL REGROUPING OF SEGMENTS AND STITCHING.

    def resize_images_to_average_width(self,images):

            average_width = int(sum(img.shape[1] for img in images) / len(images))
            resized_images = [
                cv2.resize(img, (average_width, int(round(img.shape[0] * (average_width / img.shape[1])))))
                for img in images
            ]
            return resized_images

    def concatenate_images_vertically(self,images):

        max_width = max(img.shape[1] for img in images)
        resized_images = [cv2.copyMakeBorder(img, 0, 0, 0, max_width - img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for img in images]

        return np.vstack(resized_images)

    def process_images_vertically(self,images, output_dir):


        if not isinstance(images, list) or not all(isinstance(img, np.ndarray) for img in images):
            raise ValueError("The “images” parameter must be a list of image matrices (OpenCV).")

        resized_images = self.resize_images_to_average_width(images)

        result = self.concatenate_images_vertically(resized_images)
        output_path = os.path.join(output_dir, f'{len(images) + 3}.png')
        cv2.imwrite(output_path, result)
        return result

    def adjust_image_portion(image, x, target_width, original_width):

        height, width = image.shape[:2]
        x = int(round(x))
        adjustment_width = int(round(target_width - (original_width - x)))

        if adjustment_width > 0:

            extended_part = np.zeros((height, adjustment_width, 3), dtype=image.dtype)
            adjusted_image = np.hstack((image[:, :x], extended_part, image[:, x:]))
        elif adjustment_width < 0:
            new_width = width + adjustment_width
            adjusted_image = np.hstack((image[:, :x], image[:, x:new_width]))
        else:
            adjusted_image = image

        return adjusted_image

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

        return valid_segments, result_vertical

    def is_null_image(self,img):
        if img is None:
            return True
        if not isinstance(img, np.ndarray):
            return True
        if img.size == 0:
            return True
        if img.max() == img.min():
            return True
        return False


    def process_and_concatenate_images_vertically(self,images, output_dir):
        for i, img in enumerate(images):
            print(f"Image {i}: Type={type(img)}, Shape={getattr(img, 'shape', None)}")
        if not isinstance(images, list) or not all(isinstance(img, np.ndarray) for img in images):
            raise ValueError("The “images” parameter must be a list of image matrices (OpenCV).")

        average_width = int(sum(img.shape[1] for img in images) / len(images))

        resized_images = [
            cv2.resize(img, (average_width, int(round(img.shape[0] * (average_width / img.shape[1])))))
            for img in images
        ]

        max_width = max(img.shape[1] for img in resized_images)

        padded_images = [
            cv2.copyMakeBorder(img, 0, 0, 0, max_width - img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for img in resized_images
        ]
        result = np.vstack(padded_images)
        output_path = os.path.join(output_dir, f'{len(images) + 3}.png')
        cv2.imwrite(output_path, result)

        return result

    def resize_image_portion(self,image, x, target_portion_width):

        height, width = image.shape[:2]
        x = int(round(x))
        target_portion_width = int(round(target_portion_width))
        if x >= width:
            return image
        left_part = image[:, :x]
        right_part = image[:, x:]
        resized_right_part = cv2.resize(right_part, (target_portion_width, height))
        adjusted_image = np.hstack((left_part, resized_right_part))
        return adjusted_image
     ##fILTERING POINTS TO AVOID BAD CASES##
    def filter_lists(self,L1, L2):
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

    def remove_duplicates(self,L1, L2):
        i = 0
        while i < len(L1) - 1:
            if L1[i][0] == L1[i + 1][0]:
                del L1[i + 1]
                del L2[i + 1]
            else:
                i += 1
        return L1, L2

    def sort_and_filter(self,L1, L2):

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
    #### DETERMINATION OF THE INTENSITIES OF EACH KEYPOINTS
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

    def filter_and_match(self,differences):
         #### This function return the matches with sensively the same intensity of brightness

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

    def index_longest_list(self,lists):
        if not lists:
            return None

        longest_index = 0
        longest_length = len(lists[0])

        for index, sublist in enumerate(lists):
            if len(sublist) > longest_length:
                longest_length = len(sublist)
                longest_index = index

        return longest_index

###########
new_curveconcatenationline= CurveConcatenationLine()
##########

def distribution_score(pts, mask, img_shape, grid_size=3):
    """
    Calculate an inlier distribution score:

      Dispersion (normalized variance in X and Y)

      Image coverage via a grid
    """
    inliers = pts[mask]
    if len(inliers) < 3:
        return 0.0

    h, w = img_shape[:2]
    #Normalized variance
    var_x = np.var(inliers[:,0]) / (w**2)
    var_y = np.var(inliers[:,1]) / (h**2)
    dispersion = (var_x + var_y)

    # Grid occupancy
    gx, gy = grid_size, grid_size
    cells = set()
    for x, y in inliers:
        cx = int(min(gx-1, gx * x / w))
        cy = int(min(gy-1, gy * y / h))
        cells.add((cx, cy))
    occ_ratio = len(cells) / (gx * gy)

    # Combined score (0 → bad, 1 → very good)
    return 0.5 * dispersion + 0.5 * occ_ratio


def affine_from_pts(src, dst, img_shape, min_pts=3,
                    ransac_thresh=3.0, confidence=0.995, refine_iters=50):


    if src is None or dst is None:
        print("src/dst missing → identity.")
        return np.eye(3, dtype=np.float32), np.zeros(0, dtype=bool)

    src = np.asarray(src, np.float32).reshape(-1, 2)
    dst = np.asarray(dst, np.float32).reshape(-1, 2)
    ok = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src, dst = src[ok], dst[ok]

    n = len(src)
    if n < min_pts:
        print(f"Not enough of points ({n} < {min_pts}) → identity.")
        return np.eye(3, dtype=np.float32), np.zeros(n, dtype=bool)

    # 1) Similarity (4 DOF)
    Ap, inl_p = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=confidence,
        maxIters=refine_iters
    )
    nin_p = int(inl_p.sum()) if inl_p is not None else 0
    score_p = distribution_score(src, inl_p.ravel().astype(bool), img_shape) if inl_p is not None else 0

    # 2) Affine (6 DOF)
    Aa, inl_a = cv2.estimateAffine2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=confidence,
        maxIters=refine_iters
    )
    nin_a = int(inl_a.sum()) if inl_a is not None else 0
    score_a = distribution_score(src, inl_a.ravel().astype(bool), img_shape) if inl_a is not None else 0

    
    # --- Choix ---
    if nin_a > nin_p:
        
        A_best, inl_best = Aa, inl_a
    elif nin_p > nin_a:
      
        A_best, inl_best = Ap, inl_p
    else:
      
        if score_a > score_p:
            
            A_best, inl_best = Aa, inl_a
        else:
            
            A_best, inl_best = Ap, inl_p

    # Format 3x3
    if A_best is None or inl_best is None:
        return np.eye(3, dtype=np.float32), np.zeros(n, dtype=bool)

    M = np.eye(3, dtype=np.float32)
    M[:2, :] = A_best.astype(np.float32)
    return M, inl_best.ravel().astype(bool)


# ================== Polygons / Masks / Ramps ==================
def poly_clip_to_rect(poly, w, h):
    def inside(P, eid):
        x,y=P
        if eid==0: return x>=0
        if eid==1: return x<=w-1
        if eid==2: return y>=0
        if eid==3: return y<=h-1
    def inter(A,B,eid):
        Ax,Ay=A; Bx,By=B
        if eid==0: t=(0-Ax)/(Bx-Ax+1e-12); return (0, Ay+t*(By-Ay))
        if eid==1: t=((w-1)-Ax)/(Bx-Ax+1e-12); return (w-1, Ay+t*(By-Ay))
        if eid==2: t=(0-Ay)/(By-Ay+1e-12); return (Ax+t*(Bx-Ax), 0)
        if eid==3: t=((h-1)-Ay)/(By-Ay+1e-12); return (Ax+t*(Bx-Ax), h-1)
    P = [tuple(p) for p in poly]
    for eid in range(4):
        out=[]
        for i in range(len(P)):
            A=P[i]; B=P[(i+1)%len(P)]
            if inside(B,eid):
                if inside(A,eid): out.append(B)
                else: out.append(inter(A,B,eid)); out.append(B)
            elif inside(A,eid):
                out.append(inter(A,B,eid))
        P=out
        if not P: break
    return np.array(P, dtype=np.float32)

def mask_from_polygon_size(size_hw, poly):
    h,w = size_hw
    mask = np.zeros((h,w), dtype=np.uint8)
    if poly.size>0: cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask

def smootherstep01(t):
    return t*t*t*(t*(t*6 - 15) + 10)

def make_target_ramp(mask_tgt, band_ratio=0.10):
    H, W = mask_tgt.shape[:2]
    inside = (mask_tgt > 0).astype(np.uint8)
    if inside.max() == 0: return np.zeros((H,W), np.float32)
    diag = float(np.hypot(W, H))
    band_px = max(40, int(band_ratio * diag))
    dist_in  = cv2.distanceTransform(inside,         cv2.DIST_L2, 5).astype(np.float32)
    dist_out = cv2.distanceTransform(1 - inside,     cv2.DIST_L2, 5).astype(np.float32)
    sdf = dist_in - dist_out
    t = np.clip(sdf / float(max(1, band_px)), 0.0, 1.0)
    return smootherstep01(t).astype(np.float32)

# ================== Diagnostics & refit affine ==================
def affine_diagnostics(A, src_pts, dst_pts, Aglob=None, cell_bbox=None):
    A = np.asarray(A, np.float32)
    src = np.asarray(src_pts, np.float32).reshape(-1, 2)
    dst = np.asarray(dst_pts, np.float32).reshape(-1, 2)
    n = len(src)
    if n >= 1:
        S_h = np.hstack([src, np.ones((n,1), np.float32)])
        Ph = (S_h @ A.T)
        z = np.clip(Ph[:, 2:3], 1e-9, None)
        pred = Ph[:, :2] / z
        rmse = float(np.sqrt(np.mean(np.sum((pred - dst)**2, axis=1))))
    else:
        rmse = 1e9
    A2 = A[:2, :2]
    try:
        _, s, _ = np.linalg.svd(A2.astype(np.float32))
        cond = float(s[0] / max(s[-1], 1e-9))
    except Exception:
        cond = 1e9
    det = float(np.linalg.det(A2.astype(np.float32)))
    delta_mean = 0.0
    if Aglob is not None and cell_bbox is not None:
        x0, x1, y0, y1 = cell_bbox
        xs = np.linspace(x0, x1, num=5, dtype=np.float32)
        ys = np.linspace(y0, y1, num=5, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        P = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3)
        Minv = safe_inv(A)
        Ginv = safe_inv(np.asarray(Aglob, np.float32))
        Pl = (P @ Minv.T); zl = np.clip(Pl[:,2:3], 1e-9, None); Pl_xy = Pl[:, :2] / zl
        Pg = (P @ Ginv.T); zg = np.clip(Pg[:,2:3], 1e-9, None); Pg_xy = Pg[:, :2] / zg
        delta_mean = float(np.mean(np.linalg.norm(Pl_xy - Pg_xy, axis=1)))
    return rmse, n, det, cond, delta_mean

def fit_affine_ridge(src_pts, dst_pts, Aglob, lam=1.8, weights=None):
    src = np.asarray(src_pts, np.float32).reshape(-1,2)
    dst = np.asarray(dst_pts, np.float32).reshape(-1,2)
    n = len(src)
    if n < 3:
        M = np.eye(3, dtype=np.float32); M[:2,:] = Aglob[:2,:]; return M
    X = np.hstack([src, np.ones((n,1),np.float32)])
    if weights is None:
        W = np.eye(n, dtype=np.float32)
    else:
        w = np.clip(weights.astype(np.float32).ravel(), 1e-6, None)
        W = np.diag(w)
    XtWX = X.T @ W @ X
    I = np.eye(3, dtype=np.float32)
    Ag = Aglob[:2,:]
    A_est = np.zeros((2,3), np.float32)
    for dim in range(2):
        y = dst[:,dim:dim+1]
        ag = Ag[dim:dim+1,:].T
        lhs = XtWX + lam * I
        rhs = X.T @ W @ y + lam * ag
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        A_est[dim,:] = sol.ravel()
    M = np.eye(3, dtype=np.float32); M[:2,:] = A_est
    return M

def refit_cell_affine(cell, src_pts, dst_pts, Wt, Ht, Aglob, neigh_cells=None, lam=2.2):
    cx, cy = cell["center"]
    dst_int = np.round(dst_pts).astype(int)
    keep = (dst_int[:,0]>=0) & (dst_int[:,0]<Wt) & (dst_int[:,1]>=0) & (dst_int[:,1]<Ht)

    def collect(mask):
        idx = np.where(keep & (mask[dst_int[:,1], dst_int[:,0]] > 0))[0]
        return src_pts[idx], dst_pts[idx]

    all_sp, all_dp = [], []
    sp, dp = collect(cell["mask"]); all_sp.append(sp); all_dp.append(dp)
    if neigh_cells:
        for nc in neigh_cells:
            spn, dpn = collect(nc["mask"])
            if len(spn) > 0: all_sp.append(spn); all_dp.append(dpn)

    valid_sp = [a for a in all_sp if len(a) > 0]
    valid_dp = [a for a in all_dp if len(a) > 0]
    if len(valid_sp) == 0 or len(valid_dp) == 0:
        M = np.eye(3, dtype=np.float32); M[:2,:] = Aglob[:2,:]; return M, 0.2

    SP = np.vstack(valid_sp).astype(np.float32)
    DP = np.vstack(valid_dp).astype(np.float32)
    if len(SP) < 3:
        M = np.eye(3, dtype=np.float32); M[:2,:] = Aglob[:2,:]; return M, 0.2

    d2 = (DP[:,0]-cx)**2 + (DP[:,1]-cy)**2
    x0,x1,y0,y1 = cell["bbox"]
    sigma = 0.35 * np.hypot(max(1,x1-x0), max(1,y1-y0))
    w = np.exp(-d2/(2.0*sigma*sigma)).astype(np.float32)

    M = fit_affine_ridge(SP, DP, Aglob, lam=float(lam), weights=w)
    conf = float((w.sum() / (w.max()+1e-9)) / 20.0)
    conf = max(0.2, min(1.5, conf))
    return M, conf

# ================== Canvas & Grid ==================
def build_canvas_geometry(src_shape, tgt_shape, Aglob, Hhint=None,
                          margin_ratio=0.12, min_pad=96):
    Hs, Ws = src_shape[:2]; Ht, Wt = tgt_shape[:2]
    diag_s = math.hypot(Ws, Hs)
    pad = int(max(min_pad, margin_ratio * diag_s))
    src_poly = np.float32([[0,0],[Ws-1,0],[Ws-1,Hs-1],[0,Hs-1]])
    sp_h = np.hstack([src_poly, np.ones((4,1),dtype=np.float32)])

    xs = [0, Wt-1, Wt-1, 0]; ys = [0, 0, Ht-1, Ht-1]
    tp_g = (sp_h @ Aglob.T); tp_g = tp_g[:,:2]/np.clip(tp_g[:,2:3], 1e-9, None)
    xs += tp_g[:,0].tolist(); ys += tp_g[:,1].tolist()
    if Hhint is not None:
        tp_h = (sp_h @ Hhint.T); tp_h = tp_h[:,:2]/np.clip(tp_h[:,2:3], 1e-9, None)
        xs += tp_h[:,0].tolist(); ys += tp_h[:,1].tolist()

    minx, maxx = math.floor(np.min(xs)) - pad, math.ceil(np.max(xs)) + pad
    miny, maxy = math.floor(np.min(ys)) - pad, math.ceil(np.max(ys)) + pad
    ox = -min(0, minx); oy = -min(0, miny)
    Wc = int(maxx - minx + 1); Hc = int(maxy - miny + 1)
    return Wc, Hc, int(ox), int(oy)

def grid_cells_from_mask(mask, grid=(1,1)):
    H, W = mask.shape[:2]
    ys, xs = np.where(mask>0)
    if len(xs)==0:
        xmin, xmax, ymin, ymax = 0, W-1, 0, H-1
    else:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
    padding = 30
    xmin = max(0, xmin - padding); xmax = min(W-1, xmax + padding)
    ymin = max(0, ymin - padding); ymax = min(H-1, ymax + padding)
    gx, gy = grid
    cells = []
    for j in range(gy):
        for i in range(gx):
            x0 = int(round(xmin + (xmax-xmin+1)*i/gx))
            x1 = int(round(xmin + (xmax-xmin+1)*(i+1)/gx))
            y0 = int(round(ymin + (ymax-ymin+1)*j/gy))
            y1 = int(round(ymin + (ymax-ymin+1)*(j+1)/gy))
            cmask = np.zeros((H,W), np.uint8); cmask[y0:y1, x0:x1] = 255
            cmask = cv2.bitwise_and(cmask, mask) if mask.max()>0 else cmask
            m = cv2.moments(cmask, binaryImage=True)
            if m["m00"]>0: cx = m["m10"]/m["m00"]; cy = m["m01"]/m["m00"]
            else: cx = 0.5*(x0+x1); cy = 0.5*(y0+y1)
            cells.append({"mask": cmask, "bbox": (x0,x1,y0,y1), "center": (float(cx), float(cy))})
    return cells

# ================== FFD (lattice) field ==================
def build_field_ffd(baseH, models, cells, cell_conf,
                    Wc, Hc, ox, oy, Ht, Wt,
                    overlap_mask_tgt,
                    lattice_hw=(64,64),
                    sigma_ratio=0.30,
                    scale=0.50,
                    dmax=60.0):
    Ny, Nx = lattice_hw
    yy_l, xx_l = np.mgrid[0:Ny, 0:Nx].astype(np.float32)
    xx_t_l = (xx_l / max(1,Nx-1)) * (Wc-1)
    yy_t_l = (yy_l / max(1,Ny-1)) * (Hc-1)
    Yh_l = np.stack([xx_t_l - ox, yy_t_l - oy, np.ones_like(xx_t_l)], axis=-1)

    BaseInv = safe_inv(baseH)
    Bv = (Yh_l.reshape(-1,3) @ BaseInv.T).reshape(Ny,Nx,3)
    zb = np.clip(Bv[...,2:3], 1e-9, None)
    bx = (Bv[...,0]/zb[...,0]).astype(np.float32)
    by = (Bv[...,1]/zb[...,0]).astype(np.float32)

    centers = np.float32([c["center"] for c in cells])
    if len(centers) == 0:
        return np.zeros((Hc,Wc), np.float32), np.zeros((Hc,Wc), np.float32), np.zeros((Hc,Wc), np.float32)

    cx = centers[:,0][:,None,None]
    cy = centers[:,1][:,None,None]
    dist2 = (xx_t_l[None,:,:] - cx)**2 + (yy_t_l[None,:,:] - cy)**2

    cell_diags = np.array([math.hypot(c["bbox"][1]-c["bbox"][0], c["bbox"][3]-c["bbox"][2]) for c in cells], np.float32)
    sigma = sigma_ratio * (cell_diags.mean() + 1e-6)

    w_raw = np.exp(-dist2 / (2.0 * (sigma**2)))
    conf = np.clip(np.asarray(cell_conf, np.float32), 0.2, None)[:,None,None]
    w_raw *= conf
    Wsum = np.clip(w_raw.sum(axis=0, keepdims=True), 1e-6, None)
    w_norm = w_raw / Wsum

    dx_l = np.zeros((Ny,Nx), np.float32)
    dy_l = np.zeros((Ny,Nx), np.float32)
    Yh_flat = Yh_l.reshape(-1,3)

    for k, Mk in enumerate(models):
        Minv = safe_inv(Mk)
        Lv = (Yh_flat @ Minv.T).reshape(Ny,Nx,3)
        zl = np.clip(Lv[...,2:3], 1e-9, None)
        xk = (Lv[...,0]/zl[...,0]).astype(np.float32)
        yk = (Lv[...,1]/zl[...,0]).astype(np.float32)
        dx_l += w_norm[k] * (xk - bx)
        dy_l += w_norm[k] * (yk - by)

    np.clip(dx_l, -dmax, dmax, out=dx_l)
    np.clip(dy_l, -dmax, dmax, out=dy_l)


    dx_l = cv2.GaussianBlur(dx_l, (0,0), 0.6)
    dy_l = cv2.GaussianBlur(dy_l, (0,0), 0.6)

    ramp_tgt = make_target_ramp(overlap_mask_tgt, band_ratio=0.10)
    ramp_canvas = np.zeros((Hc, Wc), np.float32)
    y0, y1 = oy, oy + Ht; x0, x1 = ox, ox + Wt
    if y0 < Hc and x0 < Wc and y1 > 0 and x1 > 0:
        yy0 = max(0, y0); yy1 = min(Hc, y1)
        xx0 = max(0, x0); xx1 = min(Wc, x1)
        ry0 = yy0 - oy; ry1 = yy1 - oy
        rx0 = xx0 - ox; rx1 = xx1 - ox
        ramp_canvas[yy0:yy1, xx0:xx1] = ramp_tgt[ry0:ry1, rx0:rx1]

    dx_full = cv2.resize(dx_l, (Wc, Hc), interpolation=cv2.INTER_CUBIC)
    dy_full = cv2.resize(dy_l, (Wc, Hc), interpolation=cv2.INTER_CUBIC)

    dx_full = (scale * dx_full * ramp_canvas).astype(np.float32)

    return dx_full, dy_full, ramp_canvas

def safe_inv(M):
    try:
        return np.linalg.inv(np.asarray(M, np.float32)).astype(np.float32)
    except Exception:
        return np.eye(3, dtype=np.float32)
def make_match_density_canvas(dst_pts, Wc, Hc, ox, oy, sigma_px=24):
    den = np.zeros((Hc, Wc), np.float32)
    if dst_pts is None or len(dst_pts)==0:
        return den
    pts = np.asarray(dst_pts, np.float32)
    xs = np.clip(np.round(ox + pts[:,0]).astype(int), 0, Wc-1)
    ys = np.clip(np.round(oy + pts[:,1]).astype(int), 0, Hc-1)
    den[ys, xs] += 1.0
    k = int(max(15, 6*int(max(1, sigma_px))));  k |= 1
    den = cv2.GaussianBlur(den, (k,k), sigma_px)
    m = den.max()
    if m > 1e-9: den /= m
    return den

def apply_seam_guard(dx_full, dy_full, ramp_canvas, dst_pts, Wc, Hc, ox, oy,
                     min_gate=0.30, blur_sigma=1.1):
    taper = smootherstep01(np.clip(ramp_canvas, 0.0, 1.0)) ** 1.2
    density = make_match_density_canvas(dst_pts, Wc, Hc, ox, oy, sigma_px=24)
    gate = min_gate + (1.0 - min_gate) * smootherstep01(np.clip(density, 0.0, 1.0))
    mask = (taper * gate).astype(np.float32)
    dx = cv2.GaussianBlur((dx_full * mask).astype(np.float32), (0,0), blur_sigma)
    dy = cv2.GaussianBlur((dy_full * mask).astype(np.float32), (0,0), blur_sigma)
    return dx, dy, mask, density

def warp_images_with_xfeat_points(src_img, tgt_img, mkpts_0, mkpts_1):
    """ here is the function which will use all the others functions in the aim to proceed to our warping"""
    # Validation
    if src_img is None or tgt_img is None:
        raise ValueError("Images invalides")
    if len(mkpts_0) < 12 or len(mkpts_1) < 12:
        raise ValueError("Pas assez de points matches (minimum 12)")

    Hs, Ws = src_img.shape[:2]
    Ht, Wt = tgt_img.shape[:2]
    src_pts = np.asarray(mkpts_0, dtype=np.float32)
    dst_pts = np.asarray(mkpts_1, dtype=np.float32)

    # =====warping base : AFFINE =====
    Aglob, inliers_mask = affine_from_pts(src_pts, dst_pts,src_img.shape)


    src_pts = src_pts[inliers_mask]
    dst_pts = dst_pts[inliers_mask]

    # -- overlap via baseH (affine)
    BaseInv = safe_inv(Aglob)
    src_poly = np.float32([[0,0],[Ws-1,0],[Ws-1,Hs-1],[0,Hs-1]])
    src_poly_w = cv2.perspectiveTransform(src_poly.reshape(-1,1,2), Aglob).reshape(-1,2)
    poly_tgt = poly_clip_to_rect(src_poly_w.tolist(), Wt, Ht)
    overlap_mask_tgt = mask_from_polygon_size((Ht,Wt), poly_tgt)

    if len(poly_tgt) > 0:
        poly_src = cv2.perspectiveTransform(poly_tgt.reshape(-1,1,2), BaseInv).reshape(-1,2)
        overlap_mask_src = mask_from_polygon_size(src_img.shape[:2], poly_src)
    else:
        overlap_mask_src = np.zeros(src_img.shape[:2], dtype=np.uint8)

    cells = grid_cells_from_mask(overlap_mask_tgt, grid=(2,2))

    # --  Local AFfine MODEL + confidence
    models, cell_conf = [], []
    dst_int_all = np.round(dst_pts).astype(int)
    for ci, c in enumerate(cells):
        cmask = c["mask"]
        keep = (dst_int_all[:,0]>=0) & (dst_int_all[:,0]<Wt) & (dst_int_all[:,1]>=0) & (dst_int_all[:,1]<Ht)
        keep &= (cmask[dst_int_all[:,1], dst_int_all[:,0]] > 0)
        sp = src_pts[keep]; dp = dst_pts[keep]

        neigh = [c2 for cj, c2 in enumerate(cells) if cj != ci]
        Mk, conf = refit_cell_affine(c, src_pts, dst_pts, Wt, Ht, Aglob, neigh_cells=neigh, lam=2.2)

        rmse, n, det, cond, dmean = affine_diagnostics(Mk, sp, dp, Aglob=Aglob, cell_bbox=c["bbox"])
        bad = (n < 6) or (rmse > 5.0) or (abs(det) < 0.05) or (cond > 120.0) or (dmean > 0.22*np.hypot(Wt,Ht)/4.0)
        if bad:
            Mk2, conf2 = refit_cell_affine(c, src_pts, dst_pts, Wt, Ht, Aglob, neigh_cells=neigh, lam=2.8)
            rmse2, n2, det2, cond2, dmean2 = affine_diagnostics(Mk2, sp, dp, Aglob=Aglob, cell_bbox=c["bbox"])
            score1 = rmse + 0.01*cond + 0.5*max(0.0,0.1-abs(det)) + 0.002*dmean
            score2 = rmse2 + 0.01*cond2 + 0.5*max(0.0,0.1-abs(det2)) + 0.002*dmean2
            if score2 < score1: Mk, conf = Mk2, conf2

        models.append(Mk.astype(np.float32))
        cell_conf.append(conf)

    # -- canvas
    Wc, Hc, ox, oy = build_canvas_geometry(src_img.shape, tgt_img.shape, Aglob, Hhint=None,
                                           margin_ratio=0.12, min_pad=96)

    # -- FIELS FFD
    dx_full, dy_full, ramp_canvas = build_field_ffd(
        Aglob, models, cells, cell_conf,
        Wc, Hc, ox, oy, Ht, Wt,
        overlap_mask_tgt,
        lattice_hw=(64,64),
        sigma_ratio=0.35,
        scale=0.50,
        dmax=60.0
    )

    # -- SeamGuard
    dx_full, dy_full, mask_seam, density = apply_seam_guard(
        dx_full, dy_full, ramp_canvas, dst_pts, Wc, Hc, ox, oy,
        min_gate=0.30, blur_sigma=1.1
    )

    # --  final map
    yy_c, xx_c = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    xx_t = xx_c - ox; yy_t = yy_c - oy
    Yh = np.stack([xx_t, yy_t, np.ones_like(xx_t)], axis=-1)
    BaseInvY = (Yh.reshape(-1,3) @ BaseInv.T).reshape(Hc,Wc,3)
    z0 = np.clip(BaseInvY[...,2:3], 1e-9, None)
    map0_x = (BaseInvY[...,0]/z0[...,0]).astype(np.float32)
    map0_y = (BaseInvY[...,1]/z0[...,0]).astype(np.float32)

    map_x = (map0_x + dx_full).astype(np.float32)
    map_y = (map0_y + dy_full).astype(np.float32)

    # ============ POINT EXTRACTION TRANSFORMED IN THE CANVAS ============
    listA = []
    listB = []

    # --- target Points  → canvasB : simple translation
    for pt in dst_pts:
        x, y = pt
        xc = x + ox
        yc = y + oy
        if 0 <= xc < Wc and 0 <= yc < Hc:
            listB.append([float(xc), float(yc)])
        else:
            listB.append([float('nan'), float('nan')])

    # ---  source points → canvasA : Aglob + FFD + SeamGuard
    src_pts_h = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=np.float32)])
    src_pts_t = (src_pts_h @ Aglob.T)
    z = np.clip(src_pts_t[:, 2], 1e-9, None)
    src_pts_proj = src_pts_t[:, :2] / z[:, None]
    src_pts_c = src_pts_proj + np.array([[ox, oy]], dtype=np.float32)

    for i, (xc, yc) in enumerate(src_pts_c):
        if not (0 <= xc < Wc and 0 <= yc < Hc):
            listA.append([float('nan'), float('nan')])
            continue

        x0, y0 = int(xc), int(yc)
        x1, y1 = min(x0 + 1, Wc - 1), min(y0 + 1, Hc - 1)

        dx00, dx01 = dx_full[y0, x0], dx_full[y0, x1]
        dx10, dx11 = dx_full[y1, x0], dx_full[y1, x1]
        dy00, dy01 = dy_full[y0, x0], dy_full[y0, x1]
        dy10, dy11 = dy_full[y1, x0], dy_full[y1, x1]

        wx, wy = xc - x0, yc - y0

        dx = (dx00 * (1 - wx) + dx01 * wx) * (1 - wy) + (dx10 * (1 - wx) + dx11 * wx) * wy
        dy = (dy00 * (1 - wx) + dy01 * wx) * (1 - wy) + (dy10 * (1 - wx) + dy11 * wx) * wy

        final_x = xc + dx
        final_y = yc + dy

        listA.append([float(final_x), float(final_y)])

    # -- warp final
    canvasA = cv2.remap(src_img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    canvasB = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    y0, y1 = oy, oy + Ht; x0, x1 = ox, ox + Wt
    if y0 < Hc and x0 < Wc and y1 > 0 and x1 > 0:
        yy0 = max(0, y0); yy1 = min(Hc, y1)
        xx0 = max(0, x0); xx1 = min(Wc, x1)
        src_y0 = yy0 - oy; src_y1 = yy1 - oy
        src_x0 = xx0 - ox; src_x1 = xx1 - ox
        canvasB[yy0:yy1, xx0:xx1] = tgt_img[src_y0:src_y1, src_x0:src_x1]

    return canvasA, canvasB, listA, listB

def safe_extract_transformed_points(data_list, index_list):
    """
   Extracts point pairs [(x1, y1), (x2, y2)] from a group in data_list,
  accepting list, tuple, and numpy.ndarray as valid containers for both groups and points.
  Malformed elements are ignored.
    """
    if not data_list:
        return []
    try:
        target_list = data_list[index_list]
    except (IndexError, TypeError):
        return []

    if not target_list:
        return []

    transformed = []
    ignored_count = 0

    seq_types = (list, tuple, np.ndarray)

    def _has_two_coords(pt):

        if not isinstance(pt, seq_types):
            return False
        try:

            size = (pt.size if isinstance(pt, np.ndarray) else len(pt))
            return size >= 2
        except Exception:
            return False

    def _xy_as_float(pt):

        arr = np.asarray(pt).reshape(-1)  # gère (2,), (1,2), etc.
        x, y = float(arr[0]), float(arr[1])
        return x, y


    for item in target_list:

        if not isinstance(item, seq_types) or len(item) < 2:
            ignored_count += 1
            continue

        pt1, pt2 = item[0], item[1]

        if not _has_two_coords(pt1):
            ignored_count += 1
            continue
        if not _has_two_coords(pt2):
            ignored_count += 1
            continue

        try:
            x1, y1 = _xy_as_float(pt1)
            x2, y2 = _xy_as_float(pt2)
            transformed.append([(x1, y1), (x2, y2)])
        except Exception:
            ignored_count += 1
            continue

    return transformed


def pair_points_list(flat_list):

    if len(flat_list) % 2 != 0:

        flat_list = flat_list[:-1]
    return [ [flat_list[i], flat_list[i+1]] for i in range(0, len(flat_list), 2) ]


#### RESEARCH OF THE FREE PARALLAX ZONE####

def research_adequate_zone_from_script(mkpts_0, mkpts_1, group_index=-1):
  category1, category2, category3, category4, category5, category6, category7,category8,category9,category10,category11,category12,category13,category14,category15,category16,category17,category18,category19,category20, summoy, nbre =0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0
  yaxis_class=[[0,100],[100,200],[200,300],[300,400],[400,500],[500,600],[600,700],[700,800],[800,900],[900,1000],[1000,1080]]
  xaxis_class=[[0,100],[100,200],[200,300],[300,400],[400,500],[500,600],[600,700],[700,800],[800,900],[900,950],[950,1000],[1000,1100],[1100,1200],[1200,1300],[1300,1400],[1400,1500],[1500,1600],[1600,1700],[1700,1800],[1800,1900],[1900,2000]]
  sum_result=None
  class_avgs=[]
  fixed_point=[]
  average_xaxis=[]
  average_yaxis=[]
  filtered_setleft,filtered_setright=[],[]
  k=0
  i,avg_abs,avg_ord1, avg_ord2,ydiff,bad_frame=0,0,0,0,0,0
  area,ydiff_left=[],[]
  concatpoint, mean_ratio=None,None
  set_points={}
  transformed=[]
  points_class,intensities_class=[],[]

  yleft_right,yright_left=0,0
  for y in range(len(mkpts_0)):
    if mkpts_0[y][0]>mkpts_1[y][0]:
      if mkpts_0[y][1]>mkpts_1[y][1]:
        filtered_setleft.append([mkpts_0[y],mkpts_1[y]])
        ydiff_left.append(mkpts_0[y][1]-mkpts_1[y][1])
        yleft_right+=1
      elif mkpts_0[y][1]<mkpts_1[y][1]:
        filtered_setright.append([mkpts_0[y],mkpts_1[y]])
        ydiff+=mkpts_1[y][1]-mkpts_0[y][1]
        yright_left+=1

  ydiff=ydiff/yright_left
  ydiff_left=statistics.mean(ydiff_left)

  filtered_setleft=[m for m in filtered_setleft if m[0][1]-m[1][1]<ydiff_left]
  filtered_setright=[m for m in filtered_setright if  m[1][1]-m[0][1]<ydiff]

  if yleft_right>yright_left:
    liste1 = [ [pair[0][0], pair[1][0]] for pair in filtered_setleft ]
    liste2 = [ [pair[0][1], pair[1][1]] for pair in filtered_setleft ]
    for elt in xaxis_class:
      xdiff=[]
      class_points=[]
      summoy1,summoy,nbre=0,0,0
      for m,n in filtered_setleft:
        if m[0]>= elt[0] and m[0]<elt[1] :
          xdiff.append(m[0]-n[0])

      try:
        avgxdiff=statistics.mean(xdiff)
      except statistics.StatisticsError:
        avgxdiff=0
      average_xaxis.append([avgxdiff,len(xdiff)])

    data = [sublist[0] for sublist in average_xaxis if sublist[-1]>5]
    gen_avg=statistics.mean(data)

    diff_avg=[]
    for j in range(len(average_xaxis)-1):
      if average_xaxis[j][1]!=0 and average_xaxis[j+1][1]!=0:
        diff_avg.append([abs(average_xaxis[j][0] -average_xaxis[j+1][0]),j])

    n=len(data)
    var= sum((x - gen_avg) ** 2 for x in data) / n
    ecart=math.sqrt(var)

    small_diff,succ=[], ecart/2

    while len(small_diff)==0:
      small_diff=[num for num in diff_avg if num[0]<=succ and average_xaxis[num[1]][1]>5]
      succ+=5
    result = []
    current_group = []
    for element in small_diff:
      if not current_group or abs(element[1] - current_group[-1][1]) <= 1:
        current_group.append(element)
      else:
        result.append(current_group)
        current_group = [element]
    if current_group:
      result.append(current_group)
    diff_genavg=[]
    area= [[] for _ in range(len(result))]

    for e in range(len(result)):

      area[e].append(xaxis_class[result[e][0][1]][0])
      area[e].append(xaxis_class[result[e][-1][1]+1][1])

    diff_genavg=[]
    if len(area)>1:
      filtered_set=[[] for _ in range(len(result))]
      for i in range (len(area)):
        data=[m[0]-n[0] for m,n in filtered_setleft if area[i][0] <= m[0] <= area[i][1]]
        n = len(data)
        class_avg = statistics.mean(data)
        var= sum((x - class_avg) ** 2 for x in data) / n
        ecart=math.sqrt(var)

        diff_genavg.append([area[i],abs(class_avg-gen_avg)/n,[class_avg,ecart]])
        diff_genavg=sorted(diff_genavg, key=lambda x:x[1] )
      area=[m[0] for m in diff_genavg ]
      filtered_setcopy=[]
      for i in range (len(area)):
        error=diff_genavg[i][-1][1]/2
        class_avg=diff_genavg[i][-1][0]
        while len(filtered_set[i])==0 and error<diff_genavg[i][-1][1]/2+10:
          #if gen_avg-150 < class_avg < gen_avg+150:
          filtered_set[i]=[[m,n,m[0]-n[0]-class_avg] for m,n in filtered_setleft if area[i][0] <= m[0] <= area[i][1] and class_avg-error <m[0]-n[0]<class_avg+error]
          error+=5
        if len(filtered_set[i])>0:
          filtered_set[i]=sorted(filtered_set[i], key=lambda x:x[-1] )
          filtered_set[i]=[elt[:2] for elt in filtered_set[i]]
          filtered_setcopy.append(filtered_set[i])
          class_avgs.append(class_avg)
          set_points[str(area[i])]=filtered_set[i]
      try:
        mean_ratio=statistics.mean([class_avgs[0]-m for m in class_avgs[1:] if class_avgs[0]-m>0 and class_avgs[0]-m <=100])
      except statistics.StatisticsError:
        mean_ratio=None
      try:
        concatpoint=filtered_setcopy[0][int(len(filtered_setcopy)/2)][:2]


        filtered_setcopy_for_detection = [item for sublist in filtered_setcopy for item in sublist]

        transformed = safe_extract_transformed_points(filtered_setcopy, group_index)

      except IndexError:
        print("pas de points")

    else:
      filtered_set=[]
      data=[m[0]-n[0] for m,n in filtered_setleft if area[0][0] <= m[0] <= area[0][1]]
      n = len(data)
      class_avg = statistics.mean(data)
      var= sum((x - class_avg) ** 2 for x in data) / n
      ecart=math.sqrt(var)
      error=ecart/2
      while error<ecart/2+10:
        filtered_set=[[m,n,m[0]-n[0]-class_avg] for m,n in filtered_setleft if area[0][0] <= m[0] <= area[0][1] and class_avg-error <m[0]-n[0]<class_avg+error]
        error+=5

      class_avgs.append(class_avg)
      set_points[str(area[0])]=filtered_set

      try:
        filtered_set=sorted(filtered_set, key=lambda x:x[-1] )
        concatpoint=filtered_set[int(len(filtered_set)/2)][:2]

        transformed = [
            [(float(m[0]), float(m[1])), (float(n[0]), float(n[1]))]
            for (m, n, _) in filtered_set
        ]

      except IndexError:
        print("pas de points")

  else:

    for elt in xaxis_class:
      xdiff=[]

      summoy1,summoy,nbre=0,0,0
      for m,n in filtered_setright:
        if elt[0]<= m[0]<elt[1] :

          xdiff.append(m[0]-n[0])
      try:
        avgxdiff=statistics.mean(xdiff)
        average_xaxis.append([avgxdiff,len(xdiff)])
      except statistics.StatisticsError:
        avgxdiff=0
        average_xaxis.append([avgxdiff,len(xdiff)])
    data = [sublist[0] for sublist in average_xaxis if sublist[-1]>5]
    gen_avg=statistics.mean(data)

    diff_avg=[]
    for j in range(len(average_xaxis)-1):
      if average_xaxis[j][1]!=0 and average_xaxis[j+1][1]!=0:
        diff_avg.append([abs(average_xaxis[j][0] -average_xaxis[j+1][0]),j])

    small_diff,succ=[], 10

    while len(small_diff)==0:
      small_diff=[num for num in diff_avg if num[0]<=succ and average_xaxis[num[1]][1]>5]
      succ+=5

    copysmall_diff=copy.deepcopy(small_diff)
    result = []
    current_group = []
    for element in small_diff:
      if not current_group or abs(element[1] - current_group[-1][1]) <= 1:
        current_group.append(element)
      else:
        result.append(current_group)
        current_group = [element]
    if current_group:
      result.append(current_group)

    area= [[] for _ in range(len(result))]
    diff_genavg=[]

    for e in range(len(result)):

      area[e].append(xaxis_class[result[e][0][1]][0])
      area[e].append(xaxis_class[result[e][-1][1]+1][1])

    diff_genavg=[]
    if len(area)>1:
      filtered_set=[[] for _ in range(len(result))]

      for i in range (len(area)):
        data=[m[0]-n[0] for m,n in filtered_setright if area[i][0] <= m[0] <= area[i][1]]
        n = len(data)
        class_avg = statistics.mean(data)
        var= sum((x - class_avg) ** 2 for x in data) / n
        ecart=math.sqrt(var)

        diff_genavg.append([area[i],abs(class_avg-gen_avg)/n,[class_avg,ecart]])
        diff_genavg=sorted(diff_genavg, key=lambda x:x[1] )
      area=[m[0] for m in diff_genavg ]
      class_avgs=[]
      filtered_setcopy=[]
      for i in range (len(area)):
        error=diff_genavg[i][-1][1]/2
        class_avg=diff_genavg[i][-1][0]

        while len(filtered_set[i])==0 and error<diff_genavg[i][-1][1]/2+10:
          #if gen_avg-150<class_avg<gen_avg+150:
          filtered_set[i]=[[m,n,m[0]-n[0]-class_avg] for m,n in filtered_setright if area[i][0] <= m[0] <= area[i][1] and class_avg-error <m[0]-n[0]<class_avg+error]
          error+=5
        if len(filtered_set[i])>0:
          filtered_set[i]=sorted(filtered_set[i], key=lambda x:x[-1] )
          filtered_set[i]=[elt[:2] for elt in filtered_set[i]]
          filtered_setcopy.append(filtered_set[i])
          class_avgs.append(class_avg)
          set_points[str(area[i])]=filtered_set[i]

      try:
        mean_ratio=statistics.mean([class_avgs[0]-m for m in class_avgs[1:] if class_avgs[0]-m>0 and class_avgs[0]-m <=100])
      except statistics.StatisticsError:
        mean_ratio=None

      try:

        concatpoint=filtered_setcopy[0][int(len(filtered_setcopy)/2)][:2]

        #transformed = [[(point[0], point[1])] for sublist in filtered_setcopy[0] for point in sublist]
        transformed = safe_extract_transformed_points(filtered_setcopy, group_index)

      except IndexError:
        print("pas de points")

    else:
      filtered_set=[]
      data=[m[0]-n[0] for m,n in filtered_setright if area[0][0] <= m[0] <= area[0][1]]
      n = len(data)
      class_avg = statistics.mean(data)
      var= sum((x - class_avg) ** 2 for x in data) / n
      ecart=math.sqrt(var)
      error=ecart/2
      while len(filtered_set)==0 and error<ecart/2+10:
        filtered_set=[[m,n,m[0]-n[0]-class_avg] for m,n in filtered_setright if area[0][0] <= m[0] <= area[0][1] and class_avg-error <m[0]-n[0]<class_avg+error]
        error+=5

      class_avgs.append(class_avg)
      set_points[str(area[0])]=filtered_set
      try:
        filtered_set=sorted(filtered_set, key=lambda x:x[-1] )
        concatpoint=filtered_set[int(len(filtered_set)/2)][:2]


        transformed  =transformed = [
            [(float(m[0]), float(m[1])), (float(n[0]), float(n[1]))]
            for (m, n, _) in filtered_set
        ]

      except IndexError:
        print("pas de points")
    return transformed

def research_wrapper(mkpts_0, mkpts_1, result_queue):

        try:

            transformed = research_adequate_zone_from_script(mkpts_0, mkpts_1)
            result_queue.put(("SUCCESS", transformed))
        except Exception as e:
            result_queue.put(("ERROR", str(e)))

def research_with_real_timeout(mkpts_0, mkpts_1, timeout=5):

        result_queue = multiprocessing.Queue()

        p = multiprocessing.Process(
            target=research_wrapper,
            args=(mkpts_0, mkpts_1, result_queue)
        )

        p.start()
        p.join(timeout=timeout)

        if p.is_alive():

            p.terminate()
            p.join()
            return None

        if not result_queue.empty():
            status, result = result_queue.get()
            if status == "SUCCESS":
                return result
            else:

                return None
        else:

            return None
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


def _center_crop(img, H, W):
    h, w = img.shape[:2]
    y0 = max((h - H) // 2, 0)
    x0 = max((w - W) // 2, 0)
    return img[y0:y0+H, x0:x0+W]

def _crop_all_to_min(*arrs):
    H = min(a.shape[0] for a in arrs)
    W = min(a.shape[1] for a in arrs)
    return tuple(_center_crop(a, H, W) for a in arrs)

###Compute PSNR and SSIM on overlap's masked regions

def psnr_ssim_with_warp_mask(result_image, reference_image, warp_mask, data_range=None):
    """
    Compute PSNR and SSIM on overlap's masked regions .
    """
    # Align to common size
    min_h = min(result_image.shape[0], reference_image.shape[0], warp_mask.shape[0])
    min_w = min(result_image.shape[1], reference_image.shape[1], warp_mask.shape[1])
    result_image = result_image[:min_h, :min_w].astype(np.float32, copy=False)
    reference_image = reference_image[:min_h, :min_w].astype(np.float32, copy=False)
    warp_mask = warp_mask[:min_h, :min_w]

    # Drop alpha if present
    if result_image.ndim == 3 and result_image.shape[2] == 4:
        result_image = result_image[..., :3]
    if reference_image.ndim == 3 and reference_image.shape[2] == 4:
        reference_image = reference_image[..., :3]

    # Auto data_range: prefer checking value domain rather than contrast amplitude
    if data_range is None:
        def in_01(x):
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
            return xmin >= -1e-6 and xmax <= 1.0 + 1e-6
        data_range = 1.0 if (in_01(reference_image) and in_01(result_image)) else 255.0

    # Mask -> binary HxW
    if warp_mask.ndim == 3:
        warp_mask = warp_mask[..., 0]
    if warp_mask.max() > 1.0:
        warp_mask = warp_mask / 255.0
    mask_bool = warp_mask > 0.5

    if not np.any(mask_bool):
        return float('inf'), float('nan')

    # PSNR on masked pixels
    try:
        psnr = peak_signal_noise_ratio(reference_image[mask_bool], result_image[mask_bool], data_range=data_range)
        if not np.isfinite(psnr):
            psnr = float('inf')
    except Exception:
        psnr = float('inf')

    # SSIM on bbox + mask
    try:
        rows = np.any(mask_bool, axis=1); cols = np.any(mask_bool, axis=0)
        if not np.any(rows) or not np.any(cols):
            return float(psnr), float('nan')
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        ref_crop  = reference_image[rmin:rmax+1, cmin:cmax+1]
        res_crop  = result_image[rmin:rmax+1, cmin:cmax+1]
        mask_crop = mask_bool[rmin:rmax+1, cmin:cmax+1]

        # If color with unexpected channels, fallback to luminance-ish gray
        ch_axis = 2 if (ref_crop.ndim == 3 and ref_crop.shape[2] == 3) else None
        if ref_crop.ndim == 3 and ref_crop.shape[2] not in (1, 3):
            ref_crop = ref_crop.mean(axis=2)
            res_crop = res_crop.mean(axis=2)
            ch_axis = None

        h, w = ref_crop.shape[:2]
        if h < 7 or w < 7:
            return float(psnr), float('nan')

        win_size = min(11, h, w)
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(3, win_size)

        ssim = structural_similarity(
            ref_crop, res_crop,
            channel_axis=ch_axis,
            data_range=data_range,
            win_size=win_size,
            gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False,
            mask=mask_crop
        )
        if np.isnan(ssim):
            ssim = float('nan')
    except Exception:
        ssim = float('nan')

    return float(psnr), float(ssim)



def _points_from_source(src, key):

    if isinstance(src, dict):
        return list(src.get(key, []))

    if isinstance(src, (list, tuple, np.ndarray)):
        return list(src)

    return []
def _sanitize(points,H,W):
      pts = []
      seen_y = -10**12
      for x, y in sorted(points, key=lambda p: p[1]):
          xi = max(0, min(int(round(x)), W - 1))
          yi = max(0, min(int(round(y)), H - 1))
          if yi > seen_y:
              pts.append((xi, yi))
              seen_y = yi
      return np.asarray(pts, dtype=np.int32)

def image_stitching(im1, im2):
  debut=time.time()
  im1_undistorted=im1
  im2_undistorted=im2

  print("Matches extraction ...")
  mkpts_0, mkpts_1 = xfeat.match_xfeat(im1_undistorted, im2_undistorted, top_k = 4096, min_cossim=-1)

  print("Warping of images ...")

  im1_undistorted, im2_undistorted, mkpts_0, mkpts_1 = warp_images_with_xfeat_points(im1_undistorted, im2_undistorted, mkpts_0, mkpts_1)

  height1,width1=im1_undistorted.shape[:2]

  print("Initiation of the appropriate zone selection ...")
  transformed = research_with_real_timeout(mkpts_0, mkpts_1, timeout=5)
  if transformed is None or len(transformed) == 0:

      transformed = [[p1, p2] for p1, p2 in zip(mkpts_0,mkpts_1)]

  filtered_list_transformed = []

  print("Initiation of selecting points to define the stitching line ...")

  current_list = transformed[0]
  filtered_list_transformed.append(current_list)

  for next_list in transformed[1:]:

      current_y = current_list[1][1]
      next_y = next_list[1][1]
      if next_y > current_y:
          filtered_list_transformed.append(next_list)
          current_list = next_list
  transformed=filtered_list_transformed

  matched_kp1 = [pair[0] for pair in transformed]
  matched_kp2 = [pair[1] for pair in transformed]
  intensities1,list_features_we_gets_intensity1,directions1,top_point1, bottom_point1=new_curveconcatenationline.get_scharr_intensities(im1,matched_kp1)
  intensities2,list_features_we_gets_intensity2,directions2,top_point2, bottom_point2=new_curveconcatenationline.get_scharr_intensities(im2,matched_kp2)
  differences = new_curveconcatenationline.compute_intensity_differences(intensities1, intensities2)
  matched_kp1, matched_kp2 = new_curveconcatenationline.filter_and_match(differences)
  latest_point, latest_point_corresponding,closest_left, corresponding_left, closest_top_left, corresponding_top_left, closest_right, corresponding_right,closest_bottom, corresponding_bottom,closest_bottom_right, corresponding_bottom_right =new_curveconcatenationline.find_closest_points(matched_kp1,matched_kp2)
  matched_kp1,matched_kp2=new_curveconcatenationline.remove_duplicates(matched_kp1,matched_kp2)
  matched_kp2, matched_kp1=new_curveconcatenationline.remove_duplicates(matched_kp2,matched_kp1)
  label_matched_kp1=new_curveconcatenationline.labelling_points(matched_kp1)
  label_matched_kp2=new_curveconcatenationline.labelling_points(matched_kp2)
  path_ids, path_points = new_curveconcatenationline.find_path(label_matched_kp1)
  points_img2_img1 = new_curveconcatenationline.getting_points_img2_corresponding(label_matched_kp2,path_ids)
  filtered_L2 = [points_img2_img1[0]]
  filtered_L1 = [path_points[0]]
  for i in range(1, len(points_img2_img1)):
      if filtered_L2[-1][1] < points_img2_img1[i][1]:
          filtered_L2.append(points_img2_img1[i])
          filtered_L1.append(path_points[i])
  points_img2_img1= filtered_L2
  path_points= filtered_L1

  im3=im1_undistorted.copy()
  im4=im2_undistorted.copy()
  H, W = im3.shape[:2]
  print("Selection process of the stitching line's keypoints ...")
  vertical_points1, mask1,path_points1,check1 = new_curveconcatenationline.plot_lines_on_image(im1_undistorted,
                                                                                                    points=path_points,
                                                                                                    case = "im1")

  vertical_points2, mask2,path_points2,check2 = new_curveconcatenationline.plot_lines_on_image(im2_undistorted,
                                                                                                    points=points_img2_img1,
                                                                                                    case = "im2")
  print("Refining the selection process of the stitching line's keypoints ...")
  path_points1,path_points2=new_curveconcatenationline.filter_lists(path_points1,path_points2)
  path_points1, path_points2 = new_curveconcatenationline.validate_and_filter_points_by_ratio(path_points1,path_points2, min_ratio=0.7, max_ratio=1)
  print("Horizontal slicing ...")
  segments1 = new_curveconcatenationline.horizontal_cut(im3, path_points1)

  segments2 = new_curveconcatenationline.horizontal_cut(im4, path_points2)

  list_dimensions1, list_points_adapt1 = new_curveconcatenationline.list_dimensions_list_points_adapt(segments1, path_points1)
  list_dimensions2, list_points_adapt2 = new_curveconcatenationline.list_dimensions_list_points_adapt(segments2, path_points2)

  c = 1 if im3.ndim == 2 else im3.shape[2]
  print("Horizontal assembling of segments ...")
  result_horizontals,shift_first,_,list_cut= new_curveconcatenationline.process_segments(segments1, segments2, list_points_adapt1, list_points_adapt2)

  (res1, res2) = new_curveconcatenationline.extraction_usefull_area(
        im3, im4,
        path_points1,
        path_points2,
        keep1="left",
        keep2="right"
    )
  out1, line1, mask_refine1 = res1
  out2, line2, mask_refine2 = res2

  result_horiz_images = [t[0] for t in result_horizontals if t and t[0] is not None]
  adjusted_segments, result_vertical = new_curveconcatenationline.process_segmentsV(result_horiz_images, matched_kp1)

  psnr, ssim,=psnr_ssim_with_warp_mask(result_vertical,im4,mask_refine2, data_range=255)

  result_vertical=crop_black_borders(result_vertical)
  return result_vertical,psnr, ssim

from google.colab import drive
drive.mount('/content/drive')

import os, time
import numpy as np
import cv2

def get_image_files(directory):
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith(('.jpg', '.png', '.JPG', '.PNG', '.jpeg'))]

dataset_dir = '/content/sample_data/datasets'
results_dir = '/content/sample_data/results'
os.makedirs(results_dir, exist_ok=True)

image_files = get_image_files(dataset_dir)

psnr_l = []
ssim_l = []
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
        result_vertical, psnr, ssim = image_stitching(im1, im2)
    except Exception as e:
        print(f"❌ Erreur dans image_stitching: {e}")
        continue

    # Gestion intelligente des cas PSNR = inf
    if psnr is None or (isinstance(psnr, float) and np.isnan(psnr)):
        print("⚠️ PSNR invalide (None/NaN) → ignoré.")
    elif np.isinf(psnr):
        print("⚠️ PSNR inf → on ignore PSNR et SSIM pour ce couple.")
        psnr = None
        ssim = None
    else:
        psnr_l.append(float(psnr))
        if ssim is not None and not (isinstance(ssim, float) and np.isnan(ssim)):
            ssim_l.append(float(ssim))

    print(f"   ↳ PSNR={psnr}  SSIM={ssim}")

    # Sauvegarde du résultat
    result_filename = f"stitched_{i//2 + 1}.jpg"
    cv2.imwrite(os.path.join(results_dir, result_filename), result_vertical)
    print(f"💾 Saved image : {result_filename}")

    elapsed = time.time() - start_time
    total_time += elapsed
    pairs_done += 1
    print(f"⏱️ Running time for this pair: {elapsed:.4f} seconds\n")

# ======= Récap moyennes =======
psnr_avg = float(np.mean(psnr_l)) if len(psnr_l) > 0 else float('nan')
ssim_avg = float(np.mean(ssim_l)) if len(ssim_l) > 0 else float('nan')

print("========== SUMMARY ==========")
print(f"Pairs processed         : {pairs_done}")
print(f"Average PSNR (valid)    : {psnr_avg:.3f} dB" if np.isfinite(psnr_avg) else "Average PSNR (valid)    : NaN")
print(f"Average SSIM (valid)    : {ssim_avg:.4f}"   if np.isfinite(ssim_avg) else "Average SSIM (valid)    : NaN")
if pairs_done > 0:
    print(f"Total time              : {total_time:.2f} s")
    print(f"Avg time per pair       : {total_time / pairs_done:.2f} s")
print("=============================")


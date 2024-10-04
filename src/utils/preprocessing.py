import os
import json
from pathlib import Path
import shutil

import numpy as np
import cv2



class Preprocessing:
    def __init__(
        self, 
        folder_path="data", 
        processed_path="data_processed", 
        check_path="data_check",
        box_height=120, 
        box_width=120, 
        step_y=60, 
        step_x=60
    ):
        self.folder_path = folder_path
        self.processed_path = processed_path
        self.check_path = check_path
        self.box_height = box_height
        self.box_width = box_width
        self.step_y = step_y
        self.step_x = step_x
    
    def process(self, presentationLUT="IDENTITY", verbose=False):
        global_index = 0
        random_folders = []
        check_folder = f"{self.check_path}_{presentationLUT}"
        if not os.path.exists(check_folder):
            os.makedirs(check_folder)
            folders = os.listdir(self.folder_path)
            folders_presentation = []
            for folder in folders:
                with open(f"{self.folder_path}/{folder}/info.json", "r") as f:
                    info_file = json.load(f)
                for k, v in info_file.items():
                    if v["presentationLUT"] == presentationLUT:
                        folders_presentation.append(folder)
                        break

            random_folders = np.random.choice(folders_presentation, 10, replace=False)

            for folder in random_folders:
                shutil.copytree(f"{self.folder_path}/{folder}", f"{check_folder}/{folder}")

        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

        for k, folder in enumerate(os.listdir(self.folder_path)):
            if folder in random_folders:
                continue
            files = os.listdir(f"{self.folder_path}/{folder}")
            with open(f"{self.folder_path}/{folder}/info.json", "r") as f:
                info_file = json.load(f)
            if verbose:
                print("Progress:", round(k / len(os.listdir(self.folder_path)) * 100, 2), "%")
            for file in files:
                file_path = Path(f"{self.folder_path}/{folder}/{file}")
                if file_path.suffix != ".json":
                    image = cv2.imread(file_path)

                    if info_file[file_path.stem]["presentationLUT"] != presentationLUT:
                        continue
                    
                    corner_1 = tuple(map(int, info_file[file_path.stem]["nippleRectangle"]["corner1"].values()))
                    corner_2 = tuple(map(int, info_file[file_path.stem]["nippleRectangle"]["corner2"].values()))

                    laterality = info_file[file_path.stem]["laterality"]
                    if laterality == "R":
                        image, corner_1, corner_2 = self.flip_image(image, corner_1, corner_2)
                    
                    if presentationLUT == "IDENTITY":
                        image = self.automatic_brightness_and_contrast(image)

                    positive_slices, negative_slices = self.slice_intersected_image(image, self.box_height, self.box_width, self.step_y, self.step_x, corner_1, corner_2)

                    for positive_slice in positive_slices:
                        cv2.imwrite(f"{self.processed_path}/{global_index}_1.png", positive_slice)
                        global_index += 1
                    for negative_slice in negative_slices:
                        cv2.imwrite(f"{self.processed_path}/{global_index}_0.png", negative_slice)
                        global_index += 1
        
    def delete_processed(self):
        shutil.rmtree(self.processed_path)

    def automatic_brightness_and_contrast(self, image, image_type="RGB"):
        if image_type == "RGB":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist_size = len(hist)

        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        maximum = accumulator[-1]
        clip_hist_percent = (maximum / 100.0) / 2.0

        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        alpha = 255.0 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return auto_result
    
    def flip_image(self, image, corner_1, corner_2):
        image_width = image.shape[1]

        flipped_corner_1 = (image_width - corner_1[0], corner_1[1])
        flipped_corner_2 = (image_width - corner_2[0], corner_2[1])

        return cv2.flip(image, 1), flipped_corner_1, flipped_corner_2

    def slice_intersected_image(self, image, box_height, box_width, step_y, step_x, corner_1=None, corner_2=None, return_type="images", image_type="RGB"):
        if image_type == "RGB":
            image_height, image_width, _ = image.shape
        else:
            image_height, image_width = image.shape
        
        if return_type == "images":
            image_slices_positive = []
            image_slices_negative = []
            
            for y in range(0, image_height - box_height + 1, step_y):
                for x in range(0, image_width - box_width + 1, step_x):
                    x_end = x + box_width
                    y_end = y + box_height
                    
                    image_slice = image[y:y_end, x:x_end]
                    
                    if x <= corner_1[0] <= x_end and y <= corner_1[1] <= y_end and x <= corner_2[0] <= x_end and y <= corner_2[1] <= y_end:
                        image_slices_positive.append(image_slice)
                    else:
                        image_slices_negative.append(image_slice)
            
            image_slices_positive = np.array(image_slices_positive)
            image_slices_negative = np.array(image_slices_negative)
            if len(image_slices_positive) * 4 < len(image_slices_negative):
                image_slices_negative = image_slices_negative[np.random.choice(len(image_slices_negative), len(image_slices_positive) * 4, replace=False)]
            
            return image_slices_positive, image_slices_negative

        else:
            slices = []
            
            for y in range(0, image_height - box_height + 1, step_y):
                for x in range(0, image_width - box_width + 1, step_x):
                    x_end = x + box_width
                    y_end = y + box_height
                    
                    image_slice = image[y:y_end, x:x_end]

                    slices.append(((x, y), (x_end, y_end)))
            
            return slices


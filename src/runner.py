import os
from pathlib import Path
import json
import shutil

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from src.utils.preprocessing import Preprocessing
from src.utils.utils import Utils
from src.nn.modules import NippleModel


class Runner():
    def __init__(
            self, 
            folder_check="data_check",
            presentationLUT="IDENTITY",
            device="cuda",
            imshow=True,
            verbose=False
        ):
        self.folder_check = f"{folder_check}_{presentationLUT}"
        self.presentationLUT = presentationLUT 
        self.device = device
        self.imshow = imshow
        self.verbose = verbose

        self.preprocessing = Preprocessing()
        self.utils = Utils()
        self.model = NippleModel().to(device)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(os.path.join(model_path, f"best_model_{self.presentationLUT}.pth"), weights_only=True))
        self.model.eval()

    def run(self):
        distances = []
        for folder in os.listdir(self.folder_check):
            with open(f"{self.folder_check}/{folder}/info.json", "r") as file:
                info_file = json.load(file)
            for file in os.listdir(f"{self.folder_check}/{folder}"):
                file_path = Path(f"{self.folder_check}/{folder}/{file}")
                if file_path.suffix == ".json" or info_file[file_path.stem]["presentationLUT"] != self.presentationLUT:
                    continue

                corner_1 = tuple(map(int, info_file[file_path.stem]["nippleRectangle"]["corner1"].values()))
                corner_2 = tuple(map(int, info_file[file_path.stem]["nippleRectangle"]["corner2"].values()))

                image = cv2.imread(f"{self.folder_check}/{folder}/{file}", cv2.IMREAD_GRAYSCALE)

                # Preprocessing step
                if info_file[file_path.stem]["laterality"] == "R":
                    image, corner_1, corner_2 = self.preprocessing.flip_image(image, corner_1, corner_2)
                
                if self.presentationLUT == "IDENTITY":
                    image = self.preprocessing.automatic_brightness_and_contrast(image, image_type="GRAY")
                slices = self.preprocessing.slice_intersected_image(image, 120, 120, 60, 60, corner_1=corner_1, corner_2=corner_2, return_type="slices", image_type="GRAY")

                if self.imshow:
                    image_copy = image.copy()
                    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
                
                # Model step
                slices_thresholded = []
                for slice in slices:
                    x, y = slice[0]
                    x_end, y_end = slice[1]
                    image_slice = image[y:y_end, x:x_end]
                    image_slice = Image.fromarray(image_slice)
                    image_slice = transforms.ToTensor()(image_slice).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(image_slice).squeeze()
                    if outputs >= 0.8:
                        slices_thresholded.append(slice)
                        if self.imshow:
                            image_copy = cv2.rectangle(image_copy, (x, y), (x_end, y_end), (0, 255, 0), 2)

                # Postprocessing step
                if slices_thresholded:
                    optimal_intersection, _ = self.utils.find_optimal_intersection(slices_thresholded)
                    if optimal_intersection:
                        if self.imshow:
                            image_copy = cv2.rectangle(image_copy, optimal_intersection[0], optimal_intersection[1], (255, 0, 0), 2)
                        if self.verbose:
                            image_weight = np.sqrt(image.shape[1] * image.shape[0])

                            original_center = self.utils.calculate_center([(corner_1, corner_2)])
                            optimal_center = self.utils.calculate_center([optimal_intersection])
                            print(f"Original center: {original_center}")
                            print(f"Optimal center: {optimal_center}")
                            print(f"Distance: {self.utils.distance_to_center(optimal_center, original_center) / image_weight * 100}")

                            distances.append(self.utils.distance_to_center(optimal_center, original_center) / image_weight * 100)

                if self.imshow:
                    image_copy = cv2.rectangle(image_copy, corner_1, corner_2, (0, 0, 255), 2)
                    cv2.imshow("Image", image_copy)
                    cv2.waitKey(0)
        
        if self.verbose:
            print("Mean Distance:", np.mean(distances))
    
    def delete_check_folders(self):
        shutil.rmtree(self.folder_check)

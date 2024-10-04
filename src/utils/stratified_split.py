import os
import shutil
from sklearn.model_selection import train_test_split


class StratifiedSplit:
    def __init__(
        self, 
        orig_path: str = "data_processed",
        train_path: str = "data_train",
        val_path: str = "data_val",
        val_size: float = 0.2
    ):
        self.orig_path = orig_path
        self.train_path = train_path
        self.val_path = val_path
        self.val_size = val_size

    def split(self):
        images = []
        classes = []

        for image_name in os.listdir(self.orig_path):
            class_name = image_name.split("_")[1]
            images.append(image_name)
            classes.append(class_name)
        
        train_images, val_images, _, _ = train_test_split(images, classes, stratify=classes, test_size=self.val_size)

        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)
        for image_name in train_images:
            shutil.move(os.path.join(self.orig_path, image_name), os.path.join(self.train_path, image_name))

        if not os.path.exists(self.val_path):
            os.makedirs(self.val_path)
        for image_name in val_images:
            shutil.move(os.path.join(self.orig_path, image_name), os.path.join(self.val_path, image_name))

    def delete_train_folders(self):
        shutil.rmtree(self.train_path)
        shutil.rmtree(self.val_path)

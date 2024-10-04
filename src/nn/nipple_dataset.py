import os
from torch.utils.data import Dataset
from PIL import Image

class NippleDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "data_processed",
        transform = None
    ):
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        self.transform = transform

        for image_name in os.listdir(root_dir):
            class_name = image_name[:-4].split("_")[1]
            self.images.append(os.path.join(root_dir, image_name))
            self.labels.append(int(class_name))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("L")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
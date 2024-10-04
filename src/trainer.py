import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm

from src.utils.preprocessing import Preprocessing
from src.utils.stratified_split import StratifiedSplit
from src.nn.nipple_dataset import NippleDataset
from src.nn.modules import NippleModel


class Trainer():
    def __init__(
            self, 
            root_dir="data",
            presentationLUT="BOTH",
            batch_size=64,
            epochs=1000,
            device="cuda",
            verbose=False
        ):
        assert presentationLUT in ["BOTH", "IDENTITY", "INVERSE"], "Invalid presentationLUT value provided (should be 'BOTH', 'IDENTITY' or 'INVERSE')"

        self.preprocessing = Preprocessing(folder_path=root_dir)
        self.stratified_split = StratifiedSplit(orig_path=self.preprocessing.processed_path)
        self.presentationLUT = presentationLUT
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.verbose = verbose

        self.model = NippleModel().to(self.device)

        self.eps = 1e-8

    def train(self):
        if self.presentationLUT == "BOTH":
            self.__train_presentation("IDENTITY")
            self.__train_presentation("INVERSE")
        else:
            self.__train_presentation(self.presentationLUT)
    
    def __train_presentation(self, presentationLUT):
        self.preprocessing.process(presentationLUT=presentationLUT, verbose=self.verbose)
        self.stratified_split.split()
        self.preprocessing.delete_processed()

        train_dataset = NippleDataset(root_dir=self.stratified_split.train_path, transform=transforms.ToTensor())
        val_dataset = NippleDataset(root_dir=self.stratified_split.val_path, transform=transforms.ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        opt = Adam(self.model.parameters(), lr=1e-3)

        bad_epochs = 0
        best_accuracy = -1
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(self.device), labels.to(self.device).unsqueeze(1).type(torch.float32)
                outputs = self.model(images).squeeze().unsqueeze(1).type(torch.float32)
                loss = torch.sum(-1 * (4 * labels * torch.log(outputs + self.eps) + (1 - labels) * torch.log(1 - outputs + self.eps)))
                loss.backward()
                opt.step()
                opt.zero_grad()

                running_loss += loss.item() * images.size(0)
                total += labels.size(0)
                outputs = self.__threshold_function(outputs)
                correct += outputs.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / total

            if self.verbose:
                print(f"Epoch {epoch} - Loss: {epoch_loss} - Acc: {epoch_acc}")

            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device).unsqueeze(1).type(torch.float32)
                    outputs = self.model(images).squeeze().unsqueeze(1).type(torch.float32)
                    loss = torch.sum(-1 * (4 * labels * torch.log(outputs + self.eps) + (1 - labels) * torch.log(1 - outputs + self.eps)))
                    running_loss += loss.item() * images.size(0)
                    total += labels.size(0)
                    outputs = self.__threshold_function(outputs)
                    correct += outputs.eq(labels).sum().item()
            epoch_loss = running_loss / len(val_loader.dataset)
            epoch_acc = correct / total

            if self.verbose:
                print(f"Val - Loss: {epoch_loss} - Acc: {epoch_acc}")

            if epoch_acc > best_accuracy:
                if not os.path.exists("models"):
                    os.makedirs("models")
                torch.save(self.model.state_dict(), f"models/best_model_{presentationLUT}.pth")
                best_accuracy = epoch_acc
                bad_epochs = 0
            else:
                bad_epochs += 1
            
            if bad_epochs == self.epochs // 10:
                if self.verbose:
                    print("Early stopping triggered")
                break
        
        self.stratified_split.delete_train_folders()

    def __threshold_function(self, x):
        return torch.where(x >= 0.8, torch.tensor(1), torch.tensor(0))

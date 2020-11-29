import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ANNO_PATH = 'data/age.csv'
IMG_FOLDER = 'data/CACD2000'

class ImageFilesDataset(Dataset):

    def __init__(self, img_paths, labels, transform=None,
                 target_transform=None) -> None:
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int):
        img = Image.open(self.img_paths[index])
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images = [b[0] for b in batch]
        labels = [torch.tensor([b[1]]) for b in batch]
        return torch.stack(images, dim=0), torch.stack(labels, dim=0).squeeze()


class CustomDataModule(pl.LightningDataModule):

    def __init__(self, dims, anno_path=ANNO_PATH, img_folder=IMG_FOLDER,
                 batch_size=32) -> None:
        super().__init__()
        self.dims = dims
        self.anno_path = anno_path
        self.img_folder = img_folder
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        annotations = pd.read_csv(self.anno_path, header=0, names=['id', 'label'],
                                  dtype={'id': str, 'label': int})
        annotations.id = self.img_folder + '/' + annotations.id
        self.img_paths = annotations.id.values
        self.labels = annotations.label.values
        unique_labels = np.unique(self.labels)
        self.min_label = min(unique_labels)
        self.max_label = max(unique_labels)
        self.num_classes = int(self.max_label - self.min_label + 1)
        self.labels = self.labels - self.min_label


    def setup(self, stage=None):
        if stage == None or stage == 'fit':
            self.train_paths, self.val_paths, self.train_labels, self.val_labels =\
                train_test_split(self.img_paths, self.labels, test_size=0.2, stratify=self.labels)

    def train_dataloader(self) -> DataLoader:
        trfm = T.Compose([T.Resize(self.dims),
                          T.RandomHorizontalFlip(p=0.5),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = ImageFilesDataset(self.train_paths, self.train_labels, transform=trfm)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=ImageFilesDataset.collate_fn)

    def val_dataloader(self) -> DataLoader:
        trfm = T.Compose([T.Resize(self.dims),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        val_dataset = ImageFilesDataset(self.val_paths, self.val_labels, transform=trfm)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=ImageFilesDataset.collate_fn)


    




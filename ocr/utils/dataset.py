from torch.utils.data.dataset import Dataset
import os
import json
from PIL import Image


class OCRDataset(Dataset):

    def __init__(self, root, label_file, transform=None, target_transform=None):
        self.root = root
        self.label_file = label_file
        self.transform = transform
        self.target_transform = target_transform

        with open(os.path.join(self.root, self.label_file)) as f:
            self.data = list(json.load(f).items())

    def __getitem__(self, index):
        image_name, target = self.data[index]

        img = Image.open(os.path.join(self.root, image_name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

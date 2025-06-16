import os

from torch.utils.data import Dataset
from PIL import Image


class OcclusionDetectionDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.label_file = label_file
        self.transform = transform
        with open(label_file, 'r') as f:
            lines = f.readlines()
            self.samples = [
                (line.split(',')[0].strip(), line.split(',')[1].strip()) for line in lines
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        filename, label = self.samples[item]
        img = Image.open(os.path.join(self.data_dir, filename)).convert('RGB')
        return self.transform(img), int(label)

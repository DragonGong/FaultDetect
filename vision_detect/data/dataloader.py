import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


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


class MultiCamDataset(Dataset):
    """
    Multi-Camera Dataset Loader for Occlusion Detection

    Expected directory structure:

        /root_dir/
        ├── front/
        │   ├── img001.jpg
        │   ├── img002.jpg
        │   └── labels.txt
        ├── left/
        │   ├── img001.jpg
        │   ├── img002.jpg
        │   └── labels.txt
        └── right/
            ├── img001.jpg
            ├── img002.jpg
            └── labels.txt

    Description:
        - Each subfolder (e.g., front/, left/, right/) corresponds to one camera/view.
        - All camera folders must contain:
            1. Image files with the same filenames across all views (e.g., img001.jpg).
            2. A labels.txt file with lines in the format:
                   filename.jpg,label
               where label is 0 (not occluded) or 1 (occluded).
        - The dataset loader automatically aligns samples across cameras using filenames.
        - Only image filenames that are present in **all** camera folders and label files will be used.

    Example line from labels.txt:
        img001.jpg,0
        img002.jpg,1

    Output of __getitem__:
        - imgs: Tensor of shape [N, 3, H, W], where N is number of cameras (e.g., 3)
        - labels: Tensor of shape [N], one label per camera

    Usage:
        dataset = MultiCamDataset(root_dir="/path/to/root_dir", cam_list=["front", "left", "right"])
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    """

    def __init__(self, root_dir, cam_list=None, img_size=224, transform=None):
        if cam_list is None:
            cam_list = ["front", "left", "right"]
        self.root_dir = root_dir
        self.cam_list = cam_list  # camera list
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.label_dicts = {}
        self.img_filenames = None

        for cam in cam_list:
            cam_dir = os.path.join(root_dir, cam)
            label_file = os.path.join(str(cam_dir), "labels.txt")
            label_dict = {}

            with open(label_file, "r") as f:
                for line in f:
                    fname, lbl = line.strip().split(",")
                    label_dict[fname] = int(lbl)

            self.label_dicts[cam] = label_dict

            if self.img_filenames is None:
                self.img_filenames = set(label_dict.keys())
            else:
                self.img_filenames &= set(label_dict.keys())

        self.img_filenames = sorted(list(self.img_filenames))

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        fname = self.img_filenames[idx]
        images = []
        labels = []

        for cam in self.cam_list:
            img_path = os.path.join(self.root_dir, cam, fname)
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            images.append(img)

            label = self.label_dicts[cam][fname]
            labels.append(label)

        imgs = torch.stack(images)  # [N, 3, H, W]
        labels = torch.tensor(labels, dtype=torch.long)  # [N]
        return imgs, labels

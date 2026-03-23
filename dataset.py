import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class HelmetDataset(Dataset):
    def __init__(self, root_path,image_size = 256 , transform=None, mode="train"):
        self.root_path = root_path
        self.transform = transform
        self.mode = mode
        self.image_size = image_size

        self.img_dir = os.path.join(self.root_path, mode, "images")
        self.label_dir = os.path.join(self.root_path, mode, "labels")

        self.image_files = [f for f in os.listdir(self.img_dir) if f.lower().endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(self.label_dir, base_name + ".txt")

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split(" ")

                    if len(data) == 5:
                        class_id = int(data[0])
                        box = [float(x) for x in data[1:]]

                        labels.append(class_id)
                        boxes.append(box)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
            }

            if self.transform:
                image, target = self.transform(image, target)

            return image, target



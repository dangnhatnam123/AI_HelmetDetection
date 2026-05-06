import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class HelmetDataset(Dataset):
    def __init__(self, root_path, image_size=64, mode="train"):
        self.root_path = root_path
        self.mode = mode
        self.image_size = image_size
        self.img_dir = os.path.join(self.root_path, mode, "images")
        self.label_dir = os.path.join(self.root_path, mode, "labels")

        self.samples = []

        image_files = [f for f in os.listdir(self.img_dir) if f.lower().endswith(".jpg")]

        for img_f in image_files:
            base_name = os.path.splitext(img_f)[0]
            label_p = os.path.join(self.label_dir, base_name + ".txt")

            if os.path.exists(label_p):
                with open(label_p, "r") as f:
                    for line in f:
                        if line.startswith("[source") or not line.strip(): continue
                        data = line.strip().split()
                        if len(data) == 5:
                            class_id = int(data[0])
                            if class_id in [1, 2]:
                                self.samples.append((img_f, [float(x) for x in data[1:]], class_id - 1))

        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, box, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        w_orig, h_orig = image.size

        x_c, y_c, w_b, h_b = box
        left = (x_c - w_b / 2) * w_orig
        top = (y_c - h_b / 2) * h_orig
        right = (x_c + w_b / 2) * w_orig
        bottom = (y_c + h_b / 2) * h_orig

        crop_img = image.crop((left, top, right, bottom))

        return self.transform(crop_img), torch.tensor(label, dtype=torch.long)
import torch
import torch.nn as nn


class HelmetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(HelmetClassifier, self).__init__()
        # num_classes=2 vì dataset mới chỉ trả về lớp 0 (Có mũ) và 1 (Không mũ)

        self.backbone = nn.Sequential(
            # Lớp 1: Nhận diện các cạnh và chi tiết nhỏ
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Kích thước giảm còn 32x32

            # Lớp 2: Nhận diện hình khối của mũ bảo hiểm
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Kích thước giảm còn 16x16

            # Lớp 3: Trích xuất đặc trưng sâu
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Dùng AdaptiveAvgPool để biến mọi Feature Map thành 1x1,
            # giúp mô hình không bị lỗi khi đổi kích thước ảnh đầu vào.
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Đầu ra phân loại (Classifier Head)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Chống học vẹt (Overfitting)
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Làm phẳng tensor
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn


class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes=3, image_size=256):
        super(SimpleObjectDetector, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.feature_size = self.image_size // 8
        self.flatten_size = 128 * self.feature_size * self.feature_size

        self.fc_intermediate = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier_head = nn.Linear(256, self.num_classes)
        self.regressor_head = nn.Sequential(
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(-1, self.flatten_size)
        shared_features = self.fc_intermediate(features)

        class_logits = self.classifier_head(shared_features)
        bbox_regression = self.regressor_head(shared_features)

        return class_logits, bbox_regression


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HelmetDataset
from model import HelmetClassifier


def main():
    ROOT_PATH = "./data"
    IMAGE_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = HelmetDataset(root_path=ROOT_PATH, image_size=IMAGE_SIZE, mode="train")
    valid_ds = HelmetDataset(root_path=ROOT_PATH, image_size=IMAGE_SIZE, mode="valid")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = HelmetClassifier(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_valid_acc = 0.0

    print(f"Bắt đầu huấn luyện trên: {device}")
    print(f"Tổng số mẫu huấn luyện: {len(train_ds)}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()


        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}]")
        print(f"  [TRAIN] Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  [VALID] Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(" Đã hoàn thành lưu mô hình có độ chính xác cao nhất!")
        print("-" * 40)

    print(f" Hoàn tất! Độ chính xác tốt nhất trên tập Valid: {best_valid_acc:.2f}%")


if __name__ == "__main__":
    main()
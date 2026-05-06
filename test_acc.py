import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import HelmetDataset
from model import HelmetClassifier


def main():
    ROOT_PATH = "./data"
    IMAGE_SIZE = 64
    BATCH_SIZE = 32
    MODEL_PATH = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Bắt đầu đánh giá mô hình trên thiết bị: {device}")

    try:
        test_ds = HelmetDataset(root_path=ROOT_PATH, image_size=IMAGE_SIZE, mode="test")
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Tổng số mẫu test: {len(test_ds)}")
    except Exception as e:
        print(f"Lỗi khi Dataset: {e}")
        return

    if len(test_ds) == 0:
        print("Test đang trống, vui lòng kiểm tra lại dữ liệu!")
        return

    model = HelmetClassifier(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Tải thành công trọng số từ '{MODEL_PATH}'")
    except FileNotFoundError:
        print(f"Không tìm thấy file '{MODEL_PATH}'. Hãy chạy file train trước.")
        return

    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    if test_total > 0:
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100 * test_correct / test_total

        print("-" * 50)
        print(f"----------KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ----------")
        print(f"   - Hàm mất mát : {avg_test_loss:.4f}")
        print(f"   - Độ chính xác : {test_acc:.2f}%")
        print(f"   - Chi tiết dự đoán đúng : {test_correct}/{test_total} mẫu")
        print("-" * 50)


if __name__ == "__main__":
    main()
import torch
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import HelmetClassifier  # Đảm bảo file model.py đã đổi tên class thành HelmetClassifier

# --- CẤU HÌNH ---
MODEL_PATH = "best_model.pth"
TEST_IMG_DIR = "./data/test/images"
TEST_LBL_DIR = "./data/test/labels"
IMAGE_SIZE = 64  # Phải khớp với IMAGE_SIZE lúc Train
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Co Mu", "Khong Mu"]  # Tương ứng với nhãn 0 và 1 đã xử lý trong Dataset


def test_on_data():
    # 1. Tải mô hình đã huấn luyện
    model = HelmetClassifier(num_classes=2).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Đã tải mô hình từ {MODEL_PATH}")
    else:
        print(f"❌ Không tìm thấy file {MODEL_PATH}. Hãy chạy train trước!")
        return
    model.eval()

    # 2. Transform chuẩn hóa (Giống hệt lúc Train)
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Lấy tấm ảnh đầu tiên trong thư mục test để kiểm tra
    test_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith("truoc-2109-_png.rf.59510b70ae186df43d8a06faedee58f7.jpg")]
    if not test_files:
        print("❌ Không tìm thấy ảnh trong thư mục test/images")
        return

    img_name = test_files[0]
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    lbl_path = os.path.join(TEST_LBL_DIR, os.path.splitext(img_name)[0] + ".txt")

    # 4. Đọc ảnh gốc để vẽ kết quả trực quan
    original_img = cv2.imread(img_path)
    h_orig, w_orig = original_img.shape[:2]
    img_pil = Image.open(img_path).convert("RGB")

    print(f"🔍 Đang kiểm tra ảnh: {img_name}")

    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                # Bỏ qua dòng rác trong file .txt của bạn
                if line.startswith("[source") or not line.strip(): continue

                data = line.strip().split()
                if len(data) == 5:
                    class_id = int(data[0])
                    # Chỉ test các lớp mũ (1, 2). Lớp 0 (xe máy) bỏ qua
                    if class_id in [1, 2]:
                        box = [float(x) for x in data[1:]]

                        # Giải mã tọa độ YOLO để cắt vùng ảnh
                        x_c, y_c, w_b, h_b = box
                        left = int((x_c - w_b / 2) * w_orig)
                        top = int((y_c - h_b / 2) * h_orig)
                        right = int((x_c + w_b / 2) * w_orig)
                        bottom = int((y_c + h_b / 2) * h_orig)

                        # Cắt vùng đầu
                        crop_img = img_pil.crop((left, top, right, bottom))
                        input_tensor = transform(crop_img).unsqueeze(0).to(DEVICE)

                        # AI dự đoán
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            _, predicted = torch.max(outputs, 1)
                            conf = torch.softmax(outputs, dim=1)[0][predicted].item()

                        # Vẽ khung lên ảnh gốc (Xanh: Có mũ, Đỏ: Không mũ)
                        p_idx = predicted.item()
                        label_text = f"{CLASSES[p_idx]} {conf:.2f}"
                        color = (0, 255, 0) if p_idx == 0 else (0, 0, 255)

                        cv2.rectangle(original_img, (left, top), (right, bottom), color, 2)
                        cv2.putText(original_img, label_text, (left, top - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 5. Lưu và hiển thị kết quả
    cv2.imwrite("test_result.jpg", original_img)
    print("✅ Đã hoàn tất! Mời bạn mở file 'test_result.jpg' để xem kết quả.")


if __name__ == "__main__":
    test_on_data()
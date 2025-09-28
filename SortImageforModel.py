import os
import shutil
from ultralytics import YOLO

# ==== CẤU HÌNH ====
model_path = "best.pt"            # model classification
input_folder = "test"      # folder chứa ảnh gốc
output_folder = "images_sorted"    # folder kết quả phân loại

# Load model
model = YOLO(model_path)

# Tạo folder kết quả nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Lấy danh sách ảnh
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Duyệt qua ảnh với progress bar
for file in tqdm(image_files, desc="Processing images", unit="img"):
    img_path = os.path.join(input_folder, file)

    # Predict
    results = model.predict(img_path, verbose=False)
    pred_label = results[0].names[results[0].probs.top1]

    # Tạo folder nhãn trong output
    label_folder = os.path.join(output_folder, pred_label)
    os.makedirs(label_folder, exist_ok=True)

    # Copy ảnh vào folder nhãn
    shutil.copy(img_path, os.path.join(label_folder, file))

print("✅ Done. Ảnh đã được copy vào folder kết quả theo nhãn detect.")
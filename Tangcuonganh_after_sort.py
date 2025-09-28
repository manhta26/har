import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# ==== CONFIG ====
data_dir = r"C:\Users\Admin\Downloads\HumanActionRecognition\images_sorted"   # folder chứa các folder nhãn
job_name = "transform"            # tên job để đặt prefix
num_aug = 6                  # số ảnh augment cho mỗi ảnh gốc
image_ext = (".jpg", ".jpeg", ".png")

# Augment pipeline (bao gồm resize + normalize)
augment_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Chỉ resize + normalize cho ảnh gốc
base_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Lấy toàn bộ ảnh
image_paths = []
for root, _, files in os.walk(data_dir):
    for f in files:
        if f.lower().endswith(image_ext):
            image_paths.append(os.path.join(root, f))

# Xử lý với progress bar
for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
    folder = os.path.dirname(img_path)
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Resize + normalize cho ảnh gốc
    base_img = base_transform(image=img)["image"]
    base_img = ((base_img + 1) * 127.5).clip(0, 255).astype(np.uint8)

    # Lưu lại với tên job
    new_name = f"{job_name}_{name}{ext}"
    new_path = os.path.join(folder, new_name)
    cv2.imwrite(new_path, base_img)

    # Xóa ảnh gốc cũ (tránh trùng lặp nếu cần)
    if img_path != new_path:
        os.remove(img_path)

    # Sinh ảnh augment
    for i in range(num_aug):
        augmented = augment_transform(image=img)["image"]
        aug_img = ((augmented + 1) * 127.5).clip(0, 255).astype(np.uint8)

        aug_filename = f"{job_name}_{name}_aug{i+1}{ext}"
        aug_path = os.path.join(folder, aug_filename)
        cv2.imwrite(aug_path, aug_img)

print("✅ Done. Ảnh gốc đã resize về 224x224 và augment lưu vào đúng folder.")

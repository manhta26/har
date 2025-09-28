import os
import cv2
import pandas as pd
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Đường dẫn
csv_file = "Testing_set.csv"
img_dir = "test"
output_dir = "aug_images"

# Tạo thư mục output
os.makedirs(output_dir, exist_ok=True)

# Đọc file csv
df = pd.read_csv(csv_file)

# Chia train/val (80/20)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Khai báo augmentations + resize + normalize
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Resize(224, 224),  # resize về 224x224
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # chuẩn hóa [-1,1]
])

def augment_and_save(df, split):
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        filename, label = row["filename"], row["label"]
        img_path = os.path.join(img_dir, filename)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Augment nhiều lần (train nhiều hơn val)
        n_aug = 5 if split == "train" else 1

        for i in range(n_aug):
            augmented = transform(image=img)
            aug_img = augmented["image"]

            # Convert về [0,255] để lưu
            aug_img = ((aug_img + 1) * 127.5).clip(0, 255).astype("uint8")

            # Tạo folder theo nhãn
            label_dir = os.path.join(output_dir, split, label)
            os.makedirs(label_dir, exist_ok=True)

            # Lưu ảnh
            out_name = f"{os.path.splitext(filename)[0]}_{split}_aug{i}.jpg"
            cv2.imwrite(os.path.join(label_dir, out_name),
                        cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

# Xử lý train và val
augment_and_save(train_df, "train")
augment_and_save(val_df, "val")

print("✅ Done! Dataset đã được augment + split thành train/val trong", output_dir)

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm  # progress bar

# ==== CẤU HÌNH ====
model2_path = r"C:\Users\Admin\Downloads\HumanActionRecognition\Result\train_AdamW_30\weights\best.pt"
model1_path = r"C:\Users\Admin\Downloads\HumanActionRecognition\Result\train2_SGD_30\weights\best.pt"
test_folder = r"C:\Users\Admin\Downloads\HumanActionRecognition\images_sorted"  # Folder chứa ảnh test
output_csv = "results_compare.csv"

# Load model
model1 = YOLO(model1_path)
model2 = YOLO(model2_path)

# Lấy nhãn đúng từ cấu trúc folder (giả sử test_images/class_x/*.jpg)
def get_true_label(img_path):
    return os.path.basename(os.path.dirname(img_path))

# Tạo list ảnh
image_paths = []
for root, _, files in os.walk(test_folder):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(root, f))

results_data = []

# ==== CHẠY TEST VỚI PROGRESS BAR ====
for img_path in tqdm(image_paths, desc="Testing images", unit="img"):
    true_label = get_true_label(img_path)

    for model_name, model in [("Model1", model1), ("Model2", model2)]:
        start_time = time.time()
        results = model.predict(img_path, verbose=False)
        elapsed = time.time() - start_time

        pred_label = results[0].names[results[0].probs.top1]
        score = results[0].probs.top1conf.item()
        correct = 1 if pred_label == true_label else 0
        if elapsed<1:
            results_data.append({
                "Image": os.path.basename(img_path),
                "True_Label": true_label,
                "Model": model_name,
                "Pred_Label": pred_label,
                "Score": score,
                "Correct": correct,
                "Time": elapsed
            })

# Lưu ra CSV
df = pd.DataFrame(results_data)
df.to_csv(output_csv, index=False)
print(f"✅ Saved results to {output_csv}")

# ==== VẼ BIỂU ĐỒ ====
acc = df.groupby("Model")["Correct"].mean()
time_avg = df.groupby("Model")["Time"].mean()
score_avg = df.groupby("Model")["Score"].mean()

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
acc.plot(kind="bar", color=["skyblue","salmon"])
plt.title("Accuracy")
plt.ylabel("Accuracy (%)")

plt.subplot(1,3,2)
time_avg.plot(kind="bar", color=["lightgreen","orange"])
plt.title("Average Detection Time")
plt.ylabel("Seconds")

plt.subplot(1,3,3)
score_avg.plot(kind="bar", color=["violet","gray"])
plt.title("Average Confidence Score")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

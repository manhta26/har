import cv2
import time
from ultralytics import YOLO

# Load model YOLO classification
model = YOLO(r"C:\Users\Admin\Downloads\HumanActionRecognition\runs\classify\train11\weights\best.pt")  

while True:
    path = input("Nhập path ảnh (hoặc 'exit' để thoát): ")
    if path.lower() == "exit":
        break
    path = path.replace("\\", "/")
    try:
        # Đọc ảnh từ path
        img = cv2.imread(path)
        if img is None:
            print("❌ Không đọc được ảnh, kiểm tra lại path!")
            continue

        # Bắt đầu tính thời gian detect
        start_time = time.time()

        # Predict với YOLO classification
        results = model(img)

        # Thời gian detect
        detect_time = time.time() - start_time

        # Lấy nhãn dự đoán tốt nhất
        probs = results[0].probs  # xác suất các class
        class_id = int(probs.top1)  # id class dự đoán
        score = float(probs.top1conf)  # confidence
        label = f"{model.names[class_id]} ({score:.2f})"

        # In thông tin ra console
        print(f"Ảnh: {path}")
        print(f"➡ Label: {model.names[class_id]}")
        print(f"➡ Score: {score:.4f}")
        print(f"➡ Thời gian detect: {detect_time:.4f} giây\n")

        # Vẽ nhãn + thời gian lên ảnh
        cv2.putText(img, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(img, f"{detect_time:.3f}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Hiển thị ảnh
        cv2.imshow("YOLO Classification", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Lỗi:", e)

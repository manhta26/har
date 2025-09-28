from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11s-cls.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data="datasets", epochs=30, iterations=300, optimizer="SGD", plots=False, save=False, val=False)
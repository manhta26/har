from ultralytics import YOLO


def main():
# Load sẵn model phân loại
    model = YOLO("yolo11s-cls.pt")
    model.train(data="datasets",epochs=100,imgsz=224,batch=32,max_det=1,dropout=0.2,optimizer = "SGD")
if __name__=="__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()




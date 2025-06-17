from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO ("yolov8n.yaml").load("yolov8n.pt")
    model.train(data=r"C:\Users\a\Desktop\ultralytics-main (1)\cups.yaml",imgsz=640, epochs=50,batch=16)

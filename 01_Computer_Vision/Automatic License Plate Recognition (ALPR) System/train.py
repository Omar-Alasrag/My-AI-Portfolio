from ultralytics import YOLO

def train():
    model = YOLO("yolo11n.pt") 
    
    model.train(
        data=r"data\license plate yolo dset\data.yaml",
        epochs=50,
        batch=8,
        imgsz=640,
        workers=0,
        rect=True,
        classes=[0], 
    )

if __name__ == "__main__":
    train()

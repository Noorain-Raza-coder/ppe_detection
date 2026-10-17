from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="datasets/yolo_ppe_dataset/data.yaml",
    epochs=150,
    patience=30,
    batch=-1,
    imgsz=640,
    optimizer="AdamW",
    project="runs/ppe_detection",
    name="yolo11s_baseline",
    seed=42
)
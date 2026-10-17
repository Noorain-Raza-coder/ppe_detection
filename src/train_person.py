from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
            data="/home/ai-siddiq-raza/Noorain-Docs/Personal/syook/syook_project/datasets/yolo_person_only_dataset/data.yaml",
            epochs=150,
            patience=30,
            batch=-1,
            imgsz=640,
            optimizer="AdamW",
            project="runs/person_detection",
            name="yolo11n_baseline"
        )
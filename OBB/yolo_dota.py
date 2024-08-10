from ultralytics import YOLO

model = YOLO('yolov8n-obb.pt')

# Train the model on the DOTAv2 dataset
results = model.train(data='DOTAv1.5.yaml', epochs=500, imgsz=960, batch=8)

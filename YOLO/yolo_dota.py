from ultralytics import YOLO

# Create a new YOLOv8n-OBB model from scratch
model = YOLO('yolov8n.yaml')

# Train the model on the DOTAv2 dataset
results = model.train(data='DOTAv2.yaml', epochs=100, imgsz=320)
success = model.export(format='tflite')
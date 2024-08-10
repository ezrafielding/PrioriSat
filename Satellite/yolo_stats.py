from ultralytics import YOLO

max_imgsz = 640

model = YOLO(f'./YOLO/runs/obb/img{max_imgsz}/weights/best.pt')

results = model('./datasets/DOTAv1.5/images/test', imgsz=max_imgsz, stream=True)

avg_speed = 0
for result in results:
    avg_speed = avg_speed + result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess']
avg_speed = avg_speed/937

print(f'Average Speed for imgsz {max_imgsz}: ', avg_speed)
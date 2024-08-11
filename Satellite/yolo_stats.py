import wandb
from ultralytics import YOLO

max_imgsz = 1280
device = 'CM4'

wandb.init(
    # set the wandb project where this run will be logged
    project="yolo-stats",

    # track hyperparameters and run metadata
    config={
    'max_imgsz': max_imgsz,
    'device': device
    }
)

model = YOLO(f'./YOLO/img{max_imgsz}/weights/best.pt')

results = model('../datasets/DOTAv1.5/images/test', imgsz=max_imgsz, stream=True)

avg_speed = 0
total = 0
for result in results:
    preprocess_time = result.speed['preprocess']
    inference_time = result.speed['inference']
    postprocess_time = result.speed['postprocess']

    avg_speed += preprocess_time + inference_time + postprocess_time
    total += 1

    wandb.log({
        "preprocess_time": preprocess_time,
        "inference_time": inference_time,
        "postprocess_time": postprocess_time,
        "total_time": preprocess_time + inference_time + postprocess_time
    })

avg_speed = avg_speed / total
wandb.log({"avg_speed": avg_speed})

print(f'Average Speed for imgsz {max_imgsz}: ', avg_speed)
wandb.finish()
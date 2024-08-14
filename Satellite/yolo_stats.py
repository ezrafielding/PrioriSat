import wandb
from ultralytics import YOLO
import time
import psutil
import threading

def log_usage(interval=1):
    while True:
        # Get current time
        now = time.time()
        
        # Get CPU and memory usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.used / 1024 / 1024
        cpu_freq = psutil.cpu_freq().current
        
        # Log to Weights & Biases
        wandb.log({"CPU Usage (%)": cpu_usage, "CPU Frequency (MHz)": cpu_freq, "Memory Usage (MB)": memory_usage, "Timestamp": now})
        
        # Wait for the next interval
        time.sleep(interval)

def start_wandb_logging(interval=1):
    # Create a thread for the log_usage function
    logging_thread = threading.Thread(target=log_usage, args=(interval,))
    
    # Set the thread as a daemon so it exits when the main program does
    logging_thread.daemon = True
    
    # Start the thread
    logging_thread.start()

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

start_wandb_logging(interval=1)

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
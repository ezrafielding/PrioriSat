import wandb
import open_clip
from PIL import Image
import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
import psutil
import time
import threading

base_model = 'ViT-B-16'
device = 'CM4'

def log_usage(interval=1):
    global stop_threads
    while not stop_threads:
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

wandb.init(
    # set the wandb project where this run will be logged
    project="clip-stats",

    # track hyperparameters and run metadata
    config={
    'base_model': base_model,
    'device': device
    }
)

stop_threads = False

start_wandb_logging(interval=1)

test_data = pd.read_csv('./PrioEval.csv', sep='\t')

with open(f'./CLIP/desc_{base_model}.pkl', 'rb') as file:
    text_features = pickle.load(file)

model, _, preprocess = open_clip.create_model_and_transforms(base_model, pretrained=f'./CLIP/{base_model}.pt')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active

start = time.time()
images = []
print('Load Images')
for file in tqdm(test_data['filepath'].values):
    images.append(Image.open(file))

proc_images = []
print('Preprocess')
for img in tqdm(images):
    proc_images.append(preprocess(img))

image_input = torch.tensor(np.stack(proc_images), dtype=torch.float)
print('Extract Image Features')
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
image_features /= image_features.norm(dim=-1, keepdim=True)
end = time.time()
total_proc_time = end-start
avg_proc_time = total_proc_time/len(images)
print('Calculating Similarity')
start = time.time()
similarity = text_features.numpy() @ image_features.numpy().T
end = time.time()
sim_time = end - start



wandb.run.summary["total_proc_time"] = total_proc_time
wandb.run.summary["avg_proc_time"] = avg_proc_time
wandb.run.summary["sim_time"] = sim_time

with open(f'./CLIP/sim_{base_model}_{device}.pkl', 'wb') as file:
    pickle.dump(similarity, file)

time.sleep(5)
stop_threads = True
time.sleep(2)
wandb.finish()

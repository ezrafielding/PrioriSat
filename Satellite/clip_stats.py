import wandb
import open_clip
from PIL import Image
import pandas as pd
import numpy as np
import torch
import pickle
import time
from tqdm import tqdm

base_model = 'ViT-B-16'
device = 'CM4'

wandb.init(
    # set the wandb project where this run will be logged
    project="clip-stats",

    # track hyperparameters and run metadata
    config={
    'base_model': base_model,
    'device': device
    }
)


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

wandb.finish()

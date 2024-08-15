import wandb
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import os
import pickle
from tqdm import tqdm
import numpy as np
import time
import psutil
import threading

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

class_mapping = {
    0: 'plane',
    1: 'ship',
    2: 'storage-tank',
    3: 'baseball-diamond',
    4: 'tennis-court',
    5: 'basketball-court',
    6: 'ground-track-field',
    7: 'harbor',
    8: 'bridge',
    9: 'large-vehicle',
    10: 'small-vehicle',
    11: 'helicopter',
    12: 'roundabout',
    13: 'soccer-ball-field',
    14: 'swimming-pool',
    15: 'container-crane',
    16: 'airport',
    17: 'helipad'
}

def jsonConvert(json_list, key_field):
    result_dict = {}
    for obj in json_list:
        key = obj.get(key_field)
        if key is not None:
            obj_copy = {k: v for k, v in obj.items() if k != key_field}
            result_dict[key] = obj_copy
        else:
            raise KeyError(f"Key '{key_field}' not found in JSON object: {obj}")
    return result_dict

def getImageSize(text_file_path):
    # Convert the text file path to the image path
    image_path = text_file_path.replace('./runs/obb/predict/labels/', '../datasets/PrioEval/').replace('.txt', '.jpg')
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height

def genYOLODesc(filepath):
    try:
        size_x, size_y = getImageSize(filepath)
        bb_file = pd.read_csv(filepath, sep=' ', header=None)
        counts = bb_file[0].value_counts()
        text = 'A remote sensing image containing '
        prop = []
        for label, count in counts.items():
            class_prop = {'class':class_mapping[int(label)], 'count': count}
            if count >1:
                class_info = bb_file[bb_file[0]==label].copy()
                class_info[[1,3,5,7]] = class_info[[1,3,5,7]]*size_x
                class_info[[2,4,6,8]] = class_info[[2,4,6,8]]*size_y
                class_info['x_centroid'] = class_info[[1,3,5,7]].sum(axis=1)/4
                class_info['y_centroid'] = class_info[[2,4,6,8]].sum(axis=1)/4
                avg_dist = pdist(class_info[['x_centroid', 'y_centroid']].values).mean()
                class_prop['avg_dist'] = avg_dist
                dist_text = f'with average distance {round(avg_dist,2)}px, '
            else:
                class_prop['avg_dist'] = None
                dist_text = ', '
            text = text + str(count) + ' ' + class_mapping[int(label)].replace('-', ' ') + 's ' + dist_text
            prop.append(class_prop)
        text = text[:-2] + "."
        return text, prop
    except:
        return 'A satellite image.', []

max_imgsz = 1280
device = 'CM4'

wandb.init(
    # set the wandb project where this run will be logged
    project="obb-stats",

    # track hyperparameters and run metadata
    config={
    'max_imgsz': max_imgsz,
    'device': device
    }
)

stop_threads = False
start_wandb_logging(interval=1)

model = YOLO(f'./YOLO/img{max_imgsz}.pt')

start = time.time()
print('Generating YOLO OBB')
results = model.predict('../datasets/PrioEval', imgsz=max_imgsz, save_txt=True)

pred_descriptions = []
pred_dir = './runs/obb/predict/labels/'
pred_files = [f for f in os.listdir(pred_dir)]
print('Generating Descriptions')
for f in tqdm(pred_files):
    src_path = os.path.join(pred_dir, f)
    desc, prop = genYOLODesc(src_path)
    dict = {
        'filepath': f,
        'description': desc,
        'properties': prop
    }
    pred_descriptions.append(dict)
end = time.time()
pre_process = end - start
avg_pre_process = pre_process/len(pred_files)
for desc in pred_descriptions:
    desc['filepath'] = '../datasets/PrioEval/'+(desc['filepath'].replace('.txt', '.jpg'))

with open(f"./pred{max_imgsz}.json", "w") as outfile:
    json.dump(pred_descriptions, outfile)

test_data = pd.read_csv('./PrioEval.csv', sep='\t')
pred = pd.read_json(f'./pred{max_imgsz}.json').drop('description', axis=1).rename(columns={'properties': 'pred'})
test_data = pd.merge(test_data, pred, on='filepath', how='left')

print('Calculating Similarity')
start = time.time()
y_pred = []
with open(f'./YOLO/desc.pkl', 'rb') as file:
    text_features = pickle.load(file)
for i, row in test_data.iterrows():
    try:
        json_pred = json.loads(str(row['pred']).replace("'", '"').replace("None", "0"))
        y_pred.append(jsonConvert(json_pred, 'class'))
    except:
        y_pred.append({})

similarity_mat = []
for y_p in tqdm(y_pred):
    avg_sim = []
    y_p_keys = set(y_p.keys())

    for y_t in text_features:
        y_t_keys = set(y_t.keys())

        total_sim = 0
        for key in y_t_keys:
            try:
                feats = DictVectorizer().fit_transform([y_t[key], y_p[key]])
                similarity = cosine_similarity(feats[0], feats[1])[0][0]
                total_sim += similarity
            except KeyError:
                pass
        avg_sim.append(total_sim/len(y_t_keys))
    similarity_mat.append(avg_sim)
end = time.time()
sim_time = end - start
avg_sim_time = sim_time/len(y_pred)

wandb.run.summary["preprocess"] = pre_process
wandb.run.summary["avg_preprocess"] = avg_pre_process
wandb.run.summary["calc_simularity_time"] = sim_time
wandb.run.summary["avg_simularity_time"] = avg_sim_time

with open(f'./YOLO/sim_{max_imgsz}_{device}.pkl', 'wb') as file:
    pickle.dump(np.stack(similarity_mat), file)

time.sleep(5)
stop_threads = True
time.sleep(2)
wandb.finish()

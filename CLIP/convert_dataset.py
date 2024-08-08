import json
import os
from tqdm import tqdm
import pandas as pd

file = open('../datasets/NWPU-Captions/dataset_nwpu.json', 'r')
text_data = json.load(file)

base_path = '../datasets/NWPU-Captions/NWPU_images/'

dataset = {
    "train": [],
    "test": [],
    "val": []
}

prompt = "An aerial photograph with description: "

for classname in text_data:
    print("Current class: ", classname)
    class_path = os.path.join(base_path, classname)
    for entry in tqdm(text_data[classname]):
        dataset[entry['split']].append([os.path.join(class_path, entry['filename']), prompt+entry['raw'], classname])
        dataset[entry['split']].append([os.path.join(class_path, entry['filename']), prompt+entry['raw_1'], classname])
        dataset[entry['split']].append([os.path.join(class_path, entry['filename']), prompt+entry['raw_2'], classname])
        dataset[entry['split']].append([os.path.join(class_path, entry['filename']), prompt+entry['raw_3'], classname])
        dataset[entry['split']].append([os.path.join(class_path, entry['filename']), prompt+entry['raw_4'], classname])
    
train_set = pd.DataFrame(dataset['train'], columns=["filepath", "caption", "class"])
test_set = pd.DataFrame(dataset['test'], columns=["filepath", "caption", "class"])
val_set = pd.DataFrame(dataset['val'], columns=["filepath", "caption", "class"])

train_set.to_csv('../datasets/NWPU-Captions/train.csv', index=False, sep='\t')
test_set.to_csv('../datasets/NWPU-Captions/test.csv', index=False, sep='\t')
val_set.to_csv('../datasets/NWPU-Captions/val.csv', index=False, sep='\t')


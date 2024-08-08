from skimage import segmentation, filters, color, io
from skimage import graph
import json
import os
import pickle
from tqdm import tqdm
from networkx import to_numpy_array
import numpy as np

NUMBER_OFF_NODES = 50

def get_text(text_data, filename):
    classname = filename[:-8]
    for entry in text_data[classname]:
        if entry.get("filename") == filename:
            
            return entry
        
def to_adj(g):
    adj = to_numpy_array(g)
    padding_length = ((0, NUMBER_OFF_NODES - adj.shape[0]), (0, NUMBER_OFF_NODES - adj.shape[1]))
    adj = np.pad(adj, padding_length, 'constant', constant_values=0)


def make_edge_rag(img):
    edges = filters.sobel(color.rgb2gray(img))
    labels = segmentation.slic(img, compactness=30, n_segments=NUMBER_OFF_NODES, start_label=1)
    g = graph.rag_boundary(labels, edges)

    return g
        
file = open('../datasets/NWPU-Captions/dataset_nwpu.json', 'r')
text_data = json.load(file)

base_path = '../datasets/NWPU-Captions/NWPU_images/'
test_adjs = []
test_desc = []
train_adjs = []
train_desc = []
val_adjs = []
val_desc = []

for classname in text_data:
    print("Current class: ", classname)
    class_path = os.path.join(base_path, classname)
    for entry in tqdm(text_data[classname]):
        try:
            img = io.imread(os.path.join(class_path, entry['filename']))
        except:
            continue

        g = make_edge_rag(img)
        if len(g) > NUMBER_OFF_NODES:
            edges = filters.sobel(color.rgb2gray(img))
            labels = segmentation.slic(img, compactness=30, n_segments=NUMBER_OFF_NODES-10, start_label=1)
            g = graph.rag_boundary(labels, edges)

        
        adj = to_adj(g)

        if entry['split'] == 'train':
            for i in range(5): train_adjs.append(adj)
            train_desc.append(entry['raw'])
            train_desc.append(entry['raw_1'])
            train_desc.append(entry['raw_2'])
            train_desc.append(entry['raw_3'])
            train_desc.append(entry['raw_4'])
        elif entry['split'] == 'test':
            for i in range(5): test_adjs.append(adj)
            test_desc.append(entry['raw'])
            test_desc.append(entry['raw_1'])
            test_desc.append(entry['raw_2'])
            test_desc.append(entry['raw_3'])
            test_desc.append(entry['raw_4'])
        else:
            for i in range(5): val_adjs.append(adj)
            val_desc.append(entry['raw'])
            val_desc.append(entry['raw_1'])
            val_desc.append(entry['raw_2'])
            val_desc.append(entry['raw_3'])
            val_desc.append(entry['raw_4'])

with open(os.path.join(f'../datasets/NWPU-Captions/NWPU_Graphs_{NUMBER_OFF_NODES}/train', 'graphs.pkl'), 'wb') as f:
    pickle.dump(train_adjs, f)
with open(os.path.join(f'../datasets/NWPU-Captions/NWPU_Graphs_{NUMBER_OFF_NODES}//train', 'descriptions.pkl'), 'wb') as f:
    pickle.dump(train_desc, f)

with open(os.path.join(f'../datasets/NWPU-Captions/NWPU_Graphs_{NUMBER_OFF_NODES}//test', 'graphs.pkl'), 'wb') as f:
    pickle.dump(test_adjs, f)
with open(os.path.join(f'../datasets/NWPU-Captions/NWPU_Graphs_{NUMBER_OFF_NODES}//test', 'descriptions.pkl'), 'wb') as f:
    pickle.dump(test_desc, f)

with open(os.path.join(f'../datasets/NWPU-Captions/NWPU_Graphs_{NUMBER_OFF_NODES}//dev', 'graphs.pkl'), 'wb') as f:
    pickle.dump(val_adjs, f)
with open(os.path.join(f'../datasets/NWPU-Captions/NWPU_Graphs_{NUMBER_OFF_NODES}//dev', 'descriptions.pkl'), 'wb') as f:
    pickle.dump(val_desc, f)
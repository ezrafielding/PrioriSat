import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from joblib import Parallel, delayed
from GraphGen.recognize import *

with open('../data/graphgen/graphs.pkl', 'rb') as f:
    adjs = pkl.load(f)
with open('../data/graphgen/descs.pkl', 'rb') as f:
    descs = pkl.load(f)

def _gen(adj):
    n = get_node_num(adj)
    m = get_edge_num(adj)
    max_diameter = get_max_diameter(adj)
    cc_num = get_connected_component_num(adj)
    degree_seq = get_degree_seq(adj)

    have_cycle = m > n - cc_num
    max_deg = np.max(degree_seq)
    min_deg = np.min(degree_seq)

    return n, m, max_diameter, cc_num, max_deg, min_deg, have_cycle

data = Parallel(n_jobs=-1)(delayed(_gen)(adj) for adj in tqdm(adjs))

stat = {'n': [], 'm': [], 'max_diameter': [], 'cc_num': [], 'max_deg': [], 'min_deg': [], 'have_cycle': []}
for n, m, max_diameter, cc_num, max_deg, min_deg, have_cycle in data:
    stat['n'].append(n)
    stat['m'].append(m)
    stat['max_diameter'].append(max_diameter)
    stat['cc_num'].append(cc_num)
    stat['max_deg'].append(max_deg)
    stat['min_deg'].append(min_deg)
    stat['have_cycle'].append(have_cycle)
    
df = pd.DataFrame(stat)
print(df.describe())
    
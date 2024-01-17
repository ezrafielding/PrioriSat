import pickle
import numpy as np
import random
import networkx as nx
from joblib import Parallel, delayed
from tqdm import trange, tqdm
from GraphGen.recognize import *
import os

train_tot = 100000
dev_tot = 10000
test_tot = 500
params = {
    'scale_free_graph': {
        'n': [(5, 9), (10, 24), (25, 40), (41, 50)]
    },
    'erdos_renyi_graph': {
        'np': [((5, 9), 0.3), ((10, 24), 0.2), ((25, 40), 0.1), ((41, 50), 0.05)]
    },
    'random_geometric_graph': {
        'nr': [((5, 9), 0.5), ((10, 24), 0.4), ((25, 40), 0.2), ((41, 50), 0.2)]
    },
    'random_tree': {
        'n': [(5, 9), (10, 24), (25, 40), (41, 50)]
    }
}

def nxgraph_to_adj_matrix(g: nx.graph.Graph) -> np.ndarray:
    adj = nx.adjacency_matrix(g).todense()
    # remove multi-edges
    adj[adj > 0] = 1

    # remove self-loop
    for i in range(adj.shape[0]):
        adj[i, i] = 0

    # make sure the graph is undirected
    adj = np.maximum(adj, adj.T)

    return adj

def gen_scale_free_graph(count: int) -> [np.ndarray]:
    def _gen():
        while True:
            (nl, nr) = random.choice(params['scale_free_graph']['n'])
            n = np.random.randint(nl, nr + 1)
            g = nx.scale_free_graph(n)

            # ensure that every node has at least one edge
            if min(dict(g.degree).values()) == 0:
                continue
        
            return nxgraph_to_adj_matrix(g)

    adjs = Parallel(n_jobs=8)(delayed(_gen)() for _ in tqdm(range(count)))
    
    return adjs

def gen_erdos_renyi_graph(count: int) -> [np.ndarray]:
    def _gen():
        while True:
            idx = np.random.choice(len(params['erdos_renyi_graph']['np']))
            (nl, nr), p = params['erdos_renyi_graph']['np'][idx]
            n = np.random.randint(nl, nr + 1)
            g = nx.erdos_renyi_graph(n, p)

            # ensure that every node has at least one edge
            if min(dict(g.degree).values()) == 0:
                continue
            
            return nxgraph_to_adj_matrix(g)

    adjs = Parallel(n_jobs=8)(delayed(_gen)() for _ in tqdm(range(count)))
    
    return adjs

def gen_random_geometric_graph(count: int) -> [np.ndarray]:
    def _gen():
        while True:
            idx = np.random.choice(len(params['random_geometric_graph']['nr']))
            (nl, nr), r = params['random_geometric_graph']['nr'][idx]
            n = np.random.randint(nl, nr + 1)
            g = nx.random_geometric_graph(n, r)

            # ensure that every node has at least one edge
            if min(dict(g.degree).values()) == 0:
                continue
        
            return nxgraph_to_adj_matrix(g)

    adjs = Parallel(n_jobs=8)(delayed(_gen)() for _ in tqdm(range(count)))
    
    return adjs

def gen_random_tree(count: int) -> [np.ndarray]:
    def _gen():
        while True:
            (nl, nr) = random.choice(params['random_tree']['n'])
            n = np.random.randint(nl, nr + 1)
            g = nx.random_tree(n)

            # ensure that every node has at least one edge
            if min(dict(g.degree).values()) == 0:
                continue
            
            return nxgraph_to_adj_matrix(g)

    adjs = Parallel(n_jobs=8)(delayed(_gen)() for _ in tqdm(range(count)))
    
    return adjs

def gen_properties(adjs: [np.ndarray]) -> [dict[str, int]]:
    def _gen(adj):
        n = get_node_num(adj)
        m = get_edge_num(adj)
        max_diameter = get_max_diameter(adj)
        cc_num = get_connected_component_num(adj)
        degree_seq = get_degree_seq(adj)

        have_cycle = m > n - cc_num
        max_deg = np.max(degree_seq)
        min_deg = np.min(degree_seq)

        properties = {
            'n': n,
            'm': m,
            'max_diameter': max_diameter,
            'cc_num': cc_num,
            'max_deg': max_deg,
            'min_deg': min_deg,
            'cycle': have_cycle
        }

        return properties
    
    props = Parallel(n_jobs=8)(delayed(_gen)(adj) for adj in tqdm(adjs))
    
    return props

def gen(tot: int, folder: str):
    adjs = []
    adjs.extend(gen_scale_free_graph(int(tot * 0.3)))
    adjs.extend(gen_erdos_renyi_graph(int(tot * 0.3)))
    adjs.extend(gen_random_geometric_graph(int(tot * 0.3)))
    adjs.extend(gen_random_tree(int(tot * 0.1)))

    np.random.shuffle(adjs)
    properties = gen_properties(adjs)

    with open(os.path.join(folder, 'graphs.pkl'), 'wb') as f:
        pickle.dump(adjs, f)
    with open(os.path.join(folder, 'properties.pkl'), 'wb') as f:
        pickle.dump(properties, f)

def main():
    os.makedirs('../data/graphgen/train', exist_ok=True)
    gen(train_tot, '../data/graphgen/train')
    os.makedirs('../data/graphgen/dev', exist_ok=True)
    gen(dev_tot, '../data/graphgen/dev')
    os.makedirs('../data/graphgen/test', exist_ok=True)
    gen(test_tot, '../data/graphgen/test')

if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    main()
from skimage import segmentation, filters, color, io
from skimage import graph
from GraphGen.gen import nxgraph_to_adj_matrix, gen_properties
import json
import os
import pickle
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass

def get_text(text_data, filename):
    classname = filename[:-8]
    for entry in text_data[classname]:
        if entry.get("filename") == filename:
            
            return entry
        
file = open('../datasets/NWPU-Captions/dataset_nwpu.json', 'r')
text_data = json.load(file)

base_path = '../datasets/NWPU-Captions/NWPU_images/'
test_adjs = []
test_desc = []
train_adjs = []
train_desc = []
val_adjs = []
val_desc = []

train_properties = []
test_properties = []
val_properties = []

for classname in text_data:
    print("Current class: ", classname)
    class_path = os.path.join(base_path, classname)
    for entry in text_data[classname]:
        try:
            img = io.imread(os.path.join(class_path, entry['filename']))
        except:
            continue
        edges = filters.sobel(color.rgb2gray(img))
        labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
        g = graph.rag_boundary(labels, edges)

        labels2 = graph.merge_hierarchical(labels, g, thresh=0.08, rag_copy=False,
                                        in_place_merge=True,
                                        merge_func=merge_boundary,
                                        weight_func=weight_boundary)
        while g.number_of_nodes() > 100:
            min_weight = []
            for u, v, weight in g.edges(data='weight', default=0):
                min_weight.append(weight)
            min_weight = set(min_weight)
            if len(min_weight) < 2:
                thresh = sorted(min_weight)[0] + 0.001
            else:
                thresh = sorted(min_weight)[1]
            labels2 = graph.merge_hierarchical(labels, g, thresh=thresh, rag_copy=False,
                                        in_place_merge=True,
                                        merge_func=merge_boundary,
                                        weight_func=weight_boundary)
        adj = nxgraph_to_adj_matrix(g)

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

train_properties = gen_properties(train_adjs)
test_properties = gen_properties(test_adjs)
val_properties = gen_properties(val_adjs)

with open(os.path.join('../datasets/NWPU-Captions/NWPU_Graphs_100/train', 'graphs.pkl'), 'wb') as f:
    pickle.dump(train_adjs, f)
with open(os.path.join('../datasets/NWPU-Captions/NWPU_Graphs_100/train', 'properties.pkl'), 'wb') as f:
    pickle.dump(train_properties, f)
with open(os.path.join('../datasets/NWPU-Captions/NWPU_Graphs_100/train', 'descriptions.pkl'), 'wb') as f:
    pickle.dump(train_desc, f)

with open(os.path.join('../datasets/NWPU-Captions/NWPU_Graphs_100/test', 'graphs.pkl'), 'wb') as f:
    pickle.dump(test_adjs, f)
with open(os.path.join('../datasets/NWPU-Captions/NWPU_Graphs_100/test', 'properties.pkl'), 'wb') as f:
    pickle.dump(test_properties, f)
with open(os.path.join('../datasets/NWPU-Captions/NWPU_Graphs_100/test', 'descriptions.pkl'), 'wb') as f:
    pickle.dump(test_desc, f)

with open(os.path.join('../datasets/NWPU-Captions/NWPU_Graphs_100/dev', 'graphs.pkl'), 'wb') as f:
    pickle.dump(val_adjs, f)
with open(os.path.join('../datasets/NWPU-Captions/NWPU_Graphs_100/dev', 'properties.pkl'), 'wb') as f:
    pickle.dump(val_properties, f)
with open(os.path.join('../datasets/NWPU-Captions/NWPU_Graphs_100/dev', 'descriptions.pkl'), 'wb') as f:
    pickle.dump(val_desc, f)
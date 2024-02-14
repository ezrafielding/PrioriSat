from skimage import segmentation, filters, color, io
from skimage import graph
from skimage.transform import resize
import networkx as nx
import numpy as np
import os
import time

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

# Specify the maximum dimension (either width or height)
max_dimension = 960

base_path = './datasets/DOTAv1.5/images/test'
file_list = os.listdir(base_path)
test_adjs = []
print('Total files: ', len(file_list))
count = 0
start = time.time()
for entry in file_list:
    count += 1
    try:
        img = io.imread(os.path.join(base_path, entry))
        # Get the current dimensions of the image
        height, width = img.shape[:2]

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Resize the image if either width or height exceeds the maximum dimension
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_dimension
                new_width = int(new_height * aspect_ratio)
            
            # Perform the resize operation
            img = resize(img, (new_width, new_height))

        # Now you have the resized image stored in the 'image' variable
        print(f'{count}/{len(file_list)} Converting {entry} to graph. Size: {img.shape[:2]}')
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
    test_adjs.append(adj)
end = time.time()
print('Total time: ',end-start, '\n Average time: ', (end-start)/len(file_list))
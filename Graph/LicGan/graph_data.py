import torch
from transformers import AutoTokenizer
from torch.utils import data
import pickle
import numpy as np
import os
import pickle
import inflect
p = inflect.engine()

import sys
sys.path.insert(0, '../GraphGen')
import recognize

class SyntheticGraphDataset(data.Dataset):
    """Dataset Class for synthetic graph dataset."""

    def __init__(self, data_dir, max_node, max_len, model_name='bert-base-uncased', ds_mode=0, base_seed=None):
        '''
        ds_mode 0: text with number in numeric format
        ds_mode 1: text with number in text format
        ds_mode 2: just the number in numeric format
        '''
        self.data_dir = data_dir
        with open(os.path.join(data_dir, 'graphs.pkl'), 'rb') as f:
            self.adj_matrix = pickle.load(f)
        with open(os.path.join(data_dir, 'properties.pkl'), 'rb') as f:
            self.properties = pickle.load(f)

        assert len(self.adj_matrix) == len(self.properties)

        for i in range(len(self.adj_matrix)):
            node_size = self.adj_matrix[i].shape[0]
            if node_size > max_node:
                raise Exception('Node size is larger than max_node')
            self.adj_matrix[i] = np.pad(self.adj_matrix[i], (0, max_node - node_size), 'constant', constant_values=0)
            node_inp = np.zeros((max_node,))
            node_inp[:self.properties[i]['n']] = 1
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_len = max_len
        self.max_node = max_node
        self.ds_mode = ds_mode
        self.base_seed = base_seed

    @staticmethod
    def _get_property_list(property):
        return [property['n'], property['m'], property['min_deg'], property['max_deg'], property['max_diameter'], property['cc_num'], property['cycle']]
    
    @staticmethod
    def _get_property_names(property):
        return property.keys()

    def _get_property_str_fn(self):
        return [
            lambda x: f'{x} nodes' if self.ds_mode == 0 else f'{p.number_to_words(x)} nodes' if self.ds_mode == 1 else f'{x}',
            lambda x: f'{x} edges' if self.ds_mode == 0 else f'{p.number_to_words(x)} edges' if self.ds_mode == 1 else f'{x}',
            lambda x: f'min degree {x}' if self.ds_mode == 0 else f'min degree {p.number_to_words(x)}' if self.ds_mode == 1 else f'{x}',
            lambda x: f'max degree {x}' if self.ds_mode == 0 else f'max degree {p.number_to_words(x)}' if self.ds_mode == 1 else f'{x}',
            lambda x: f'max diameter {x}' if self.ds_mode == 0 else f'max diameter {p.number_to_words(x)}' if self.ds_mode == 1 else f'{x}',
            lambda x: f'{x} connected component' if self.ds_mode == 0 else f'{p.number_to_words(x)} connected component' if self.ds_mode == 1 else f'{x}',
            lambda x: 'with cycle' if x else 'without cycle'
        ]
    
    @staticmethod
    def _get_eval_str_fn():
        # assuming g is symmetric and does not have self-loop
        return [
            lambda g, n: n.sum(), # n
            lambda g, n: g.sum() // 2, # m
            lambda g, n: g.sum(axis=0).min(), # min degree
            lambda g, n: g.sum(axis=0).max(), # max degree
            lambda g, n: recognize.get_max_diameter(g), # max diameter
            lambda g, n: recognize.get_connected_component_num(g) - (n.shape[0] - n.sum()), # cc_num
            lambda g, n: (g.sum() // 2) > (n.shape[0] - recognize.get_connected_component_num(g))   # cycle
        ]
    
    @staticmethod
    def get_prop(g, property_tuple):
        succ = []
        for i in range(len(property_tuple)):
            if property_tuple[i] is not None:
                succ.append(SyntheticGraphDataset._get_eval_str_fn()[i](g))
            else:
                succ.append(None)
        return succ

    def _gen_text(self, property, rng=None):
        property_list = self._get_property_list(property)
        # keeps node number and edges compulsorily (can change behavior by changing idx line alone)
        if rng is None:
            count = np.random.randint(2, 8) # [l, r)
            idx = [0,1] + list(np.random.choice(len(property_list) - 2, count, replace=False) + 2)
            np.random.shuffle(idx)
        else:
            count = rng.random_integers(2, 7) # [l, r]
            idx = [0,1] + list(rng.choice(len(property_list) - 2, count, replace=False) + 2)
            rng.shuffle(idx)

        text = 'Undirected graph with ' if self.ds_mode <= 1 else ''
        tag = [0] * len(property_list)
        for i in idx:
            tag[i] = 1
            text += self._get_property_str_fn()[i](property_list[i]) + ', '
        text = text[:-2] + '.'
        for i in range(len(property_list)):
            if tag[i] == 0:
                property_list[i] = None

        property_tuple = tuple(property_list)
        return text, property_tuple

    def _encode_text(self, text):
        return self.tokenizer(text, add_special_tokens=True, truncation=False, max_length=self.max_len, padding='max_length')

    def __getitem__(self, index):
        adj_matrix = self.adj_matrix[index]
        if self.base_seed is not None:
            rng = np.random.RandomState(self.base_seed + index)
        else:
            rng = None
        node_inp = np.zeros((self.max_node,))
        node_inp[:self.properties[index]['n']] = 1
        text_desc, properties = self._gen_text(self.properties[index], rng=rng)
        encoded_text = self._encode_text(text_desc)

        return adj_matrix, node_inp, encoded_text, text_desc, properties
    
    def __len__(self):
        return len(self.adj_matrix)
    
    @staticmethod
    def collate_fn(batch):
        adj_matrix = torch.from_numpy(np.stack([item[0] for item in batch])).type(torch.FloatTensor)
        node_inp = torch.from_numpy(np.stack([item[1] for item in batch])).type(torch.FloatTensor)
        ids = torch.from_numpy(np.stack([item[2].input_ids for item in batch]))
        attention_mask = torch.from_numpy(np.stack([item[2].attention_mask for item in batch]))
        desc = [item[3] for item in batch]
        properties = [item[4] for item in batch]
        return adj_matrix, node_inp, ids, attention_mask, desc, properties

def get_loaders(data_dir, max_node, max_len, model_name, batch_size, num_workers=1, ds_mode=0):
    """Build and return a data loader."""

    train = SyntheticGraphDataset(os.path.join(data_dir, 'train'), max_node, max_len, model_name, ds_mode)
    val = SyntheticGraphDataset(os.path.join(data_dir, 'dev'), max_node, max_len, model_name, ds_mode)
    test = SyntheticGraphDataset(os.path.join(data_dir, 'test'), max_node, max_len, model_name, ds_mode, base_seed=0)
    
    train_loader = data.DataLoader(dataset=train,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   collate_fn=SyntheticGraphDataset.collate_fn)
    val_loader = data.DataLoader(dataset=val,
                                 batch_size=batch_size*2,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=SyntheticGraphDataset.collate_fn)
    test_loader = data.DataLoader(dataset=test,
                                  batch_size=batch_size*2,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=SyntheticGraphDataset.collate_fn)
    return train_loader, val_loader, test_loader
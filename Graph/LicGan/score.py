import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), 'GraphGen'))

from recognize import *
import re
import numpy as np
from joblib import Parallel, delayed
from graph_data import SyntheticGraphDataset
    
def score(props, adj_mat_hat, nodes_hat):
    def _get_score(prop, adj, node):
        closeness = 0
        n_prop = []
        score_list = []
        for i, true_prop in enumerate(prop):
            eval_fn = SyntheticGraphDataset._get_eval_str_fn()[i]
            if true_prop is None:
                score_list.append(0)
                n_prop.append(0)
                continue
            
            n_prop.append(1)
            pred_prop = eval_fn(adj, node)
            score_list.append(int(int(pred_prop) == int(true_prop)))
            closeness += np.exp(-(int(pred_prop) - int(true_prop))**2)
                
        return np.sum(score_list) / np.sum(n_prop), closeness / np.sum(n_prop), score_list, n_prop

    score_close_l_n = Parallel(n_jobs=8)(delayed(_get_score)(prop, adj, node) for prop, adj, node in zip(props, adj_mat_hat, nodes_hat))
    scores, closeness, score_list, n_prop = zip(*score_close_l_n)
    return {
        'property_match': scores,
        'closeness': closeness,
        'n_match': [s[0] for s in score_list],
        'm_match': [s[1] for s in score_list],
        'min_deg_match': [s[2] for s in score_list],
        'max_deg_match': [s[3] for s in score_list],
        'diam_match': [s[4] for s in score_list],
        'cc_match': [s[5] for s in score_list],
        'cycle_match': [s[6] for s in score_list],
    }, {
        'min_deg_match': [s[2] for s in n_prop],
        'max_deg_match': [s[3] for s in n_prop],
        'diam_match': [s[4] for s in n_prop],
        'cc_match': [s[5] for s in n_prop],
        'cycle_match': [s[6] for s in n_prop],
    }

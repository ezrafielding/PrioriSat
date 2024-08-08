# Importing the libraries needed
import pandas as pd
import numpy as np
import torch
from transformers import RobertaModel
import logging
logging.basicConfig(level=logging.ERROR)

class RAGSnet(torch.nn.Module):
    def __init__(self, adj_size=10000, model_name='roberta-base'):
        super(RAGSnet, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.adj_head = torch.nn.Linear(hidden_size, adj_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        adj_matrix = self.adj_head(sequence_output)
        return adj_matrix
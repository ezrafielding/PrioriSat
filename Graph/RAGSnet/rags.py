import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import wandb

# Login to wandb
wandb.login()

# Define a custom model with RoBERTa base and an adjacency matrix head
class RAGSnet(nn.Module):
    def __init__(self, adj_size=2500, model_name='roberta-base'):
        super(RAGSnet, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.adj_head = nn.Linear(hidden_size, adj_size)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        adj_matrix = self.adj_head(sequence_output)
        return adj_matrix

# Define a custom dataset
class GraphDataset(Dataset):
    def __init__(self, texts, adjacency_matrices, tokenizer):
        self.texts = texts
        self.adjacency_matrices = adjacency_matrices
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        adj_matrix = self.adjacency_matrices[idx].ravel()
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs['label'] = torch.tensor(adj_matrix, dtype=torch.float)
        return inputs

# Instantiate the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Dummy data
with open(os.path.join('../../datasets/NWPU-Captions/NWPU_Graphs_100/train', 'graphs.pkl'), 'rb') as f:
    train_adjs = np.array(pickle.load(f))
with open(os.path.join('../../datasets/NWPU-Captions/NWPU_Graphs_100/train', 'descriptions.pkl'), 'rb') as f:
    train_desc = np.array(pickle.load(f))

# Create dataset and dataloader
dataset = GraphDataset(train_desc, train_adjs, tokenizer)

# Define the cosine similarity metric
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    # Flatten the matrices to compute cosine similarity
    labels_flat = labels.reshape(labels.shape[0], -1)
    preds_flat = preds.reshape(preds.shape[0], -1)
    
    cosine_sim = [cosine_similarity([preds_flat[i]], [labels_flat[i]])[0][0] for i in range(labels.shape[0])]
    cosine_sim_mean = np.mean(cosine_sim)
    
    return {"cosine_similarity": cosine_sim_mean}

# Initialize wandb
wandb.init(project="roberta-adjacency-matrix")

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="wandb",  # Enable wandb reporting
    run_name="roberta-adjacency-matrix-run"  # Name of the wandb run
)

# Instantiate the model
model = RAGSnet(adj_size=10000)

# # Define a custom Trainer class to use the compute_loss method
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(outputs, labels)
        return (loss, outputs) if return_outputs else loss

# Instantiate the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Finish the wandb run
wandb.finish()

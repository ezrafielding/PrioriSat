import pandas as pd
import torch
from torch import cuda
from tqdm import tqdm
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
import logging
import os
import pickle
import wandb
logging.basicConfig(level=logging.ERROR)

from model import RAGSnet
from data import GraphData

def calcuate_accuracy(preds, targets):
    cos = torch.nn.CosineSimilarity()
    return cos(preds, targets).sum().item()


def train(epoch):
    last_loss = 0
    running_loss = 0
    accuracy = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for i ,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['labels'].to(device, dtype = torch.float)

        optimizer.zero_grad()

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()

        loss.backward()
        optimizer.step()

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        running_loss += loss.items()
        if i%1000==999:
            accuracy += calcuate_accuracy(outputs, targets)
            last_loss = running_loss / 1000
            accu_step = (accuracy)/nb_tr_examples 
            print(f"Training Loss per 1000 steps: {last_loss}")
            print(f"Training Accuracy per 1000 steps: {accu_step}")

    print(f'The Total Accuracy for Epoch {epoch}: {(accuracy)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (accuracy)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    wandb.log({"epoch": epoch, "accuracy": epoch_accu, "loss": epoch_loss})

    return

def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['labels'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            n_correct += calcuate_accuracy(outputs, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct)/nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu


# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 1e-05

run = wandb.init(
    # Set the project where this run will be logged
    project="RAGSnet",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
    },
)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

device = 'cuda' if cuda.is_available() else 'cpu'

with open(os.path.join('../../datasets/NWPU-Captions/NWPU_Graphs_100/train', 'graphs.pkl'), 'rb') as f:
    train_adjs = pickle.load(f)
with open(os.path.join('../../datasets/NWPU-Captions/NWPU_Graphs_100/train', 'descriptions.pkl'), 'rb') as f:
    train_desc = pickle.load(f)

with open(os.path.join('../../datasets/NWPU-Captions/NWPU_Graphs_100/test', 'graphs.pkl'), 'rb') as f:
    test_adjs = pickle.load(f)
with open(os.path.join('../../datasets/NWPU-Captions/NWPU_Graphs_100/test', 'descriptions.pkl'), 'rb') as f:
    test_desc = pickle.load(f)

with open(os.path.join('../../datasets/NWPU-Captions/NWPU_Graphs_100/dev', 'graphs.pkl'), 'rb') as f:
    val_adjs = pickle.load(f)
with open(os.path.join('../../datasets/NWPU-Captions/NWPU_Graphs_100/dev', 'descriptions.pkl'), 'rb') as f:
    val_desc = pickle.load(f)

train_data = {
    'label': train_adjs,
    'description': train_desc
}
test_data = {
    'label': test_adjs,
    'description': test_desc
}

training_set = GraphData(pd.DataFrame(train_data), tokenizer, MAX_LEN)
testing_set = GraphData(pd.DataFrame(test_data), tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

model = RAGSnet(adj_size=10000)
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in tqdm(range(EPOCHS)):
    train(epoch)

acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)

output_model_file = 'pytorch_roberta_sentiment.bin'
output_vocab_file = './'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')
wandb.finish()

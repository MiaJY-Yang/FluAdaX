
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from bio_tokenizer import BioTokenizer
import torch
from transformers import MegaConfig, MegaForSequenceClassification
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import get_scheduler
from torch.optim import AdamW
from torch.nn import init
from tqdm.auto import tqdm
# import evaluate
from sklearn.metrics import accuracy_score, recall_score
import pickle as pkl


# real host
# data_train, data_valid = train_test_split(train_all, train_size=0.9, stratify=train_all['host'])
host2id_map = {
    'avian':0,
    'human':1
}


id2host_map = {
    0:'avian',
    1:'human'
}


dataset_path = '/home/pan/workspace/aiv/dataset.pkl'
with open(dataset_path,'rb')as file:
    dataset_all = pkl.load(file)

train = dataset_all['NS'][0]
valid = dataset_all['NS'][1]
test = dataset_all['NS'][2]

train_seq = train['seq'].to_list()
valid_seq = valid['seq'].to_list()
test_seq = test['seq'].to_list()

train_label = [host2id_map[i] for i in train['host'].to_list()]
valid_label = [host2id_map[i] for i in valid['host'].to_list()]
test_label = [host2id_map[i] for i in test['host'].to_list()]


sum(train['host']=='avian')/sum(train['host']=='human')


from bio_tokenizer import BioTokenizer
import torch
from transformers import MegaConfig, MegaForSequenceClassification
import time



# get tokenizer and model
tokenizer = BioTokenizer()


config = MegaConfig()
# update the num_vocab and num_label

config.num_labels=len(host2id_map)
config.vocab_size=10
# config.use_chunking=True
# config.chunk_size=256
# config.use_chunking=False
# config.chunk_size=-1
config.max_positions=840
config.relative_positional_bias = 'simple'
# config.num_hidden_layers=1
# config.hidden_size=128

model = MegaForSequenceClassification(config)
# model = MegaForSequenceClassification.from_pretrained('./train_0712/')


from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# train_seq_ls, train_label
# valid_seq_ls, valid_label
# test_seq_ls, test_label


class PropertyDataset(Dataset):
    def __init__(self,data_type) -> None:
        super().__init__()
        if(data_type=='train'):
            self.dataset_seq = train_seq
            self.dataset_label = train_label
        elif(data_type=='valid'):
            self.dataset_seq = valid_seq
            self.dataset_label = valid_label
        else:
            self.dataset_seq = test_seq
            self.dataset_label = test_label


    def __len__(self):
        return len(self.dataset_seq)
    
    def __getitem__(self, index):
        return self.dataset_seq[index], self.dataset_label[index]



train_data = PropertyDataset('train')
valid_data = PropertyDataset('valid')
test_data = PropertyDataset('test')


train_dataloader = DataLoader(train_data, shuffle=True, batch_size=128)
valid_dataloader = DataLoader(valid_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)



from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)



from transformers import get_scheduler

num_epochs = 50
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=num_training_steps
)



import torch
from torch.nn import init

device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)




from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_recall = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        val_recall = -val_loss
        if self.best_score is None:
            self.best_score = val_recall
            self.save_checkpoint(val_recall, model)
        elif val_recall < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_recall
            self.save_checkpoint(val_recall, model)
            self.counter = 0

    def save_checkpoint(self, val_recall, model):

        if self.verbose:
            self.trace_func(f'Validation loss increased ({self.best_recall:.6f} --> {val_recall:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        model.save_pretrained('./train_0719/')
        self.best_recall = val_recall



from tqdm import tqdm
import torch.nn as nn
num_step = 0
progress_bar = tqdm(range(num_training_steps))
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1,26.45]).to(device))


model.train()
model.to(device)
early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.005)
for epoch in range(num_epochs):
    loss_ls = []
    for batch_seq, batch_label in train_dataloader:
        batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt")
        batch_input.to(device)
        batch_label = batch_label.clone().detach().to(device=device)

        outputs = model(**batch_input)
        logits = outputs.logits
        loss = criterion(logits, batch_label)
        loss.backward()

        loss_ls.append(float(loss.clone().detach().to('cpu')))

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        num_step += 1

    print('train loss :', sum(loss_ls)/len(loss_ls))

    prediction_ls = []
    reference_ls = []
    logits_ls = []
    loss_ls_valid = []
    with torch.no_grad():
        for batch_seq, batch_label in  valid_dataloader:
            # pad_len =((max([len(i)for i in batch_seq])+1024)//1024)* 1024
            # batch_input = tokenizer(batch_seq, padding='max_length', max_length=pad_len, return_tensors="pt")
            batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt")
            batch_input.to(device=device)
            batch_label = batch_label.clone().detach().to(device=device)

            outputs = model(**batch_input)
            logits = outputs.logits

            loss = criterion(logits, batch_label)

            logits_ls.append(logits)
            loss_ls_valid.append( loss )
            predictions = torch.argmax(logits, dim=-1)
            prediction_ls += predictions.clone().detach().to('cpu').tolist()
            reference_ls += batch_label.clone().detach().to('cpu').tolist()

    acc = accuracy_score(reference_ls, prediction_ls)

    valid_macro_recall = recall_score(reference_ls, prediction_ls, average='macro')

    valid_loss = (sum(loss_ls_valid)/len(loss_ls_valid)).clone().detach().to('cpu').tolist()

    logits_all = torch.concat(logits_ls)
    logits_ls = logits_all.clone().detach().to('cpu').tolist()

    print('valid recall: ',valid_macro_recall)
    print('valid loss: ',valid_loss)

    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break



prediction_ls = []
reference_ls = []
seq_ls = []
model.to(device)
logits_ls = []
for batch_seq, batch_label in  test_dataloader:
    seq_ls.append(batch_seq)
    batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt")
    batch_input.to(device)
    batch_label = torch.tensor(batch_label).to(device)
    with torch.no_grad():
        outputs = model(**batch_input)
    logits = outputs.logits
    logits_ls += logits.clone().detach().to('cpu').tolist()

    predictions = torch.argmax(logits, dim=-1)
    prediction_ls += predictions.clone().detach().to('cpu').tolist()
    
    reference_ls += batch_label.clone().detach().to('cpu').tolist()

    # metric.add_batch(predictions=predictions, references=batch_label)
# print(metric.compute())
print('Acc: ', accuracy_score(reference_ls, prediction_ls))


save_path = 'model_simple.pth'
torch.save(model.state_dict(), save_path)



# test
from torch.nn.functional import softmax

test['test_logits'] = [softmax(torch.Tensor(i)).tolist() for i in logits_ls]
test['host_pre'] = prediction_ls_
test[test['host']=='human'][['host', 'test_logits','host_pre']]
test.to_csv('test_with_logits.csv', index=None)


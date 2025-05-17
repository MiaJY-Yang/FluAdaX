#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data_train_path = './dataset/after_2005_genome/train.csv'   
data_valid_path = './dataset/after_2005_genome/valid.csv'
data_test_path = './dataset/after_2005_genome/test.csv'

data_train = pd.read_csv(data_train_path)
data_valid = pd.read_csv(data_valid_path)
data_test = pd.read_csv(data_test_path)


# In[4]:


host2id_map = {'Human': 0,
                'Avian': 1,
                'Swine': 2,
                'Canine': 3,
                'Equine': 4}


id2host_map  = {
    0:'Human',
    1:'Avian',
    2:'Swine',
    3:'Canine',
    4:'Equine'
}


# In[5]:


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
import torch
from torch.nn import init
from tqdm.auto import tqdm
# import evaluate
from sklearn.metrics import accuracy_score, recall_score


# In[7]:


train_seq = ['<sep>'.join([ii.upper() for ii in i[4:]]) for i in data_train.values]
valid_seq = ['<sep>'.join([ii.upper() for ii in i[4:]]) for i in data_valid.values]
test_seq = ['<sep>'.join([ii.upper() for ii in i[4:]]) for i in data_test.values]


train_label = [host2id_map[i] for i in data_train['host_y'].to_list()]
valid_label = [host2id_map[i] for i in data_valid['host_y'].to_list()]
test_label = [host2id_map['Human'] if 'Human' in i else host2id_map[i] for i in data_test['host_y'].to_list()]


train_id = data_train['ID'].to_list()
valid_id = data_valid['ID'].to_list()
test_id = data_test['ID'].to_list()


# In[9]:


from bio_tokenizer import BioTokenizer
import torch
from transformers import MegaConfig, MegaForSequenceClassification
import time


# In[10]:


# get tokenizer and model
tokenizer = BioTokenizer()

config = MegaConfig()
# update the num_vocab and num_label

config.num_labels=len(host2id_map)
config.vocab_size=12
# config.use_chunking=True
# config.chunk_size=256
# config.use_chunking=False
# config.chunk_size=-1
config.max_positions=14000
config.num_hidden_layers=2
# config.hidden_size=128

model = MegaForSequenceClassification(config)
# model = MegaForSequenceClassification.from_pretrained('./train_0711/')

# model.load_state_dict(torch.load('checkpoint.pt'))


# In[11]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# In[13]:


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
            self.dataset_id = train_id
        elif(data_type=='valid'):
            self.dataset_seq = valid_seq
            self.dataset_label = valid_label
            self.dataset_id = valid_id
        else:
            self.dataset_seq = test_seq
            self.dataset_label = test_label
            self.dataset_id = test_id


    def __len__(self):
        return len(self.dataset_seq)
    
    def __getitem__(self, index):
        return self.dataset_seq[index], self.dataset_label[index],self.dataset_id[index]


# In[14]:


train_data = PropertyDataset('train')
valid_data = PropertyDataset('valid')
test_data = PropertyDataset('test')


train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8)
valid_dataloader = DataLoader(valid_data, batch_size=8)
test_dataloader = DataLoader(test_data, batch_size=8)


# In[17]:


from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=8e-5)


# In[18]:


from transformers import get_scheduler

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=num_training_steps
)


# In[19]:


import torch
from torch.nn import init

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# In[19]:


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

    def __call__(self, val_recall, model):

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
            self.trace_func(f'Validation recall increased ({self.best_recall:.6f} --> {val_recall:.6f}).  Saving model ...')
        model.save_pretrained('./train_0711/')
        self.best_recall = val_recall


# In[ ]:


from tqdm import tqdm

num_step = 0
progress_bar = tqdm(range(num_training_steps))

model.train()
model.to(device)
early_stopping = EarlyStopping(patience=4, verbose=True, delta=0.005)
for epoch in range(num_epochs):
    loss_ls = []
    for batch_seq, batch_label, batch_id in train_dataloader:
        batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt")
        batch_input.to(device)
        batch_label = batch_label.clone().detach().to(device=device)
        
        outputs = model(**batch_input, labels=batch_label)

        loss = outputs.loss
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
        for batch_seq, batch_label, batch_id in  valid_dataloader:
            # pad_len =((max([len(i)for i in batch_seq])+1024)//1024)* 1024
            # batch_input = tokenizer(batch_seq, padding='max_length', max_length=pad_len, return_tensors="pt")
            batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt")
            batch_input.to(device=device)
            batch_label = batch_label.clone().detach().to(device=device)

            outputs = model(**batch_input, labels=batch_label)
            logits = outputs.logits
            loss = outputs.loss

            logits_ls.append(logits)
            loss_ls_valid.append(loss)
            predictions = torch.argmax(logits, dim=-1)
            prediction_ls += predictions.clone().detach().to('cpu').tolist()
            reference_ls += batch_label.clone().detach().to('cpu').tolist()

    acc = accuracy_score(reference_ls, prediction_ls)
    valid_macro_recall = recall_score(reference_ls, prediction_ls, average='macro')
    valid_loss = sum(loss_ls_valid)/len(loss_ls_valid)
    logits_all = torch.concat(logits_ls)
    logits_ls = logits_all.clone().detach().to('cpu').tolist()
    print('valid acc: ',valid_macro_recall)

    # roc_auc = roc_auc_score(reference_ls, logits_ls)
    # print('ROC AUC: %.4f'%roc_auc)

    # precision, recall, thresholds = precision_recall_curve(reference_ls, logits_ls)
    # pr_auc = auc(recall, precision)
    # print(f"PR AUC: {pr_auc}")

    early_stopping(valid_macro_recall, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break


# In[20]:


prediction_ls = []
reference_ls = []
seq_ls = []
logits_ls = []
id_ls = []
model.to(device)


for batch_seq, batch_label,batch_id in  tqdm(test_dataloader):
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
    id_ls += batch_id

    # metric.add_batch(predictions=predictions, references=batch_label)
# print(metric.compute())
print('Acc: ', accuracy_score(reference_ls, prediction_ls))


# In[25]:


reference_ls_real = data_test['host_y'].tolist()


# In[27]:


avian_recall = sum([1 if reference_ls_real[i] == 'Avian' and prediction_ls_[i] == 'Avian' else 0 for i in range(len(reference_ls_real))])/sum([1 if reference_ls_real[i] == 'Avian'  else 0 for i in range(len(reference_ls_real))])
print('avian_recall: ',avian_recall, end='\t')
print(sum([1 if reference_ls_real[i] == 'Avian' and prediction_ls_[i] == 'Avian' else 0 for i in range(len(reference_ls_real))]),'/',sum([1 if reference_ls_real[i] == 'Avian'  else 0 for i in range(len(reference_ls_real))]))

canine_recall = sum([1 if reference_ls_real[i] == 'Canine' and prediction_ls_[i] == 'Canine' else 0 for i in range(len(reference_ls_real))])/sum([1 if reference_ls_real[i] == 'Canine'  else 0 for i in range(len(reference_ls_real))])
print('canine_recall: ',canine_recall, end='\t')
print(sum([1 if reference_ls_real[i] == 'Canine' and prediction_ls_[i] == 'Canine' else 0 for i in range(len(reference_ls_real))]),'/',sum([1 if reference_ls_real[i] == 'Canine'  else 0 for i in range(len(reference_ls_real))]))

equine_recall = sum([1 if reference_ls_real[i] == 'Equine' and prediction_ls_[i] == 'Equine' else 0 for i in range(len(reference_ls_real))])/sum([1 if reference_ls_real[i] == 'Equine'  else 0 for i in range(len(reference_ls_real))])
print('equine_recall: ',equine_recall, end='\t')
print(sum([1 if reference_ls_real[i] == 'Equine' and prediction_ls_[i] == 'Equine' else 0 for i in range(len(reference_ls_real))]),'/',sum([1 if reference_ls_real[i] == 'Equine'  else 0 for i in range(len(reference_ls_real))]))

human_recall = sum([1 if reference_ls_real[i] == 'Human' and prediction_ls_[i] == 'Human' else 0 for i in range(len(reference_ls_real))])/sum([1 if reference_ls_real[i] == 'Human'  else 0 for i in range(len(reference_ls_real))])
print('human_recall: ',human_recall, end='\t')
print(sum([1 if reference_ls_real[i] == 'Human' and prediction_ls_[i] == 'Human' else 0 for i in range(len(reference_ls_real))]),'/',sum([1 if reference_ls_real[i] == 'Human'  else 0 for i in range(len(reference_ls_real))]))

swine_recall = sum([1 if reference_ls_real[i] == 'Swine' and prediction_ls_[i] == 'Swine' else 0 for i in range(len(reference_ls_real))])/sum([1 if reference_ls_real[i] == 'Swine'  else 0 for i in range(len(reference_ls_real))])
print('swine_recall: ',swine_recall, end='\t')
print(sum([1 if reference_ls_real[i] == 'Swine' and prediction_ls_[i] == 'Swine' else 0 for i in range(len(reference_ls_real))]),'/',sum([1 if reference_ls_real[i] == 'Swine'  else 0 for i in range(len(reference_ls_real))]))

human_aiv_recall = sum([1 if reference_ls_real[i] == 'Human-avian' and prediction_ls_[i] == 'Human' else 0 for i in range(len(reference_ls_real))])/sum([1 if reference_ls_real[i] == 'Human-avian'  else 0 for i in range(len(reference_ls_real))])
print('human_aiv_recall: ',human_aiv_recall, end='\t')
print(sum([1 if reference_ls_real[i] == 'Human-avian' and prediction_ls_[i] == 'Human' else 0 for i in range(len(reference_ls_real))]),'/',sum([1 if reference_ls_real[i] == 'Human-avian'  else 0 for i in range(len(reference_ls_real))]))

human_siv_recall = sum([1 if reference_ls_real[i] == 'Human-swine' and prediction_ls_[i] == 'Human' else 0 for i in range(len(reference_ls_real))])/sum([1 if reference_ls_real[i] == 'Human-swine'  else 0 for i in range(len(reference_ls_real))])
print('human_siv_recall: ',human_siv_recall, end='\t')
print(sum([1 if reference_ls_real[i] == 'Human-swine' and prediction_ls_[i] == 'Human' else 0 for i in range(len(reference_ls_real))]),'/',sum([1 if reference_ls_real[i] == 'Human-swine'  else 0 for i in range(len(reference_ls_real))]))


# In[29]:


data_test['predict_host'] = prediction_ls_


# In[ ]:


from torch.nn.functional import softmax

logits_ls_norm = [softmax(torch.Tensor(i)).detach().tolist() for i in logits_ls]


# In[32]:


data_test['logits_ls_norm'] = logits_ls_norm


# In[34]:


data_test.to_csv('test_with_predict_910.csv',index=None)


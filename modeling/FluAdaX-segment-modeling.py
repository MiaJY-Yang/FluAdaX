# get data
import pandas as pd

data_train = pd.read_csv('../data/data_model/train_valid_test/train_all.csv')
data_valid = pd.read_csv('../data/data_model/train_valid_test/valid_all.csv')
data_test = pd.read_csv('../data/data_model/train_valid_test/test_all.csv')


host2id_map = {'Human': 0,
                'Swine': 1,
                'Avian': 2,
                'Canine': 3,
                'Equine': 4}


id2host_map  = {
    0:'Human',
    1:'Swine',
    2:'Avian',
    3:'Canine',
    4:'Equine'
}




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



# real host
# data_train, data_valid = train_test_split(train_all, train_size=0.9, stratify=train_all['host'])

train_seq = data_train['seq'].to_list()
valid_seq = data_valid['seq'].to_list()
test_seq = data_test['seq'].to_list()

train_label = [host2id_map[i] for i in data_train['real_host'].to_list()]
valid_label = [host2id_map[i] for i in data_valid['real_host'].to_list()]
test_label = [host2id_map[i] for i in data_test['real_host'].to_list()]




from bio_tokenizer import BioTokenizer
import torch
from transformers import MegaConfig, MegaForSequenceClassification
import time



# get tokenizer and model
tokenizer = BioTokenizer()


config = MegaConfig()
# update the num_vocab and num_label

config.num_labels=len(host2id_map)
config.vocab_size=11
# config.use_chunking=True
# config.chunk_size=256
# config.use_chunking=False
# config.chunk_size=-1
config.max_positions=2560
# config.num_hidden_layers=2
# config.hidden_size=128

model = MegaForSequenceClassification(config)
# model = MegaForSequenceClassification.from_pretrained('./model/')



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

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=num_training_steps
)




import torch
from torch.nn import init

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
        # torch.save(model.state_dict(), self.path)
        model.save_pretrained('./train_0910/')
        self.best_recall = val_recall




from tqdm import tqdm

num_step = 0
progress_bar = tqdm(range(num_training_steps))

model.train()
model.to(device)
early_stopping = EarlyStopping(patience=4, verbose=True, delta=0.005)
for epoch in range(num_epochs):
    loss_ls = []
    for batch_seq, batch_label in train_dataloader:
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
        for batch_seq, batch_label in  valid_dataloader:
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



prediction_ls = []
reference_ls = []
seq_ls = []
model.to(device)
for batch_seq, batch_label in  test_dataloader:
    seq_ls.append(batch_seq)
    batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt")
    batch_input.to(device)
    batch_label = torch.tensor(batch_label).to(device)
    with torch.no_grad():
        outputs = model(**batch_input)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    prediction_ls += predictions.clone().detach().to('cpu').tolist()
    reference_ls += batch_label.clone().detach().to('cpu').tolist()

    # metric.add_batch(predictions=predictions, references=batch_label)
# print(metric.compute())
print('Acc: ', accuracy_score(reference_ls, prediction_ls))



# [avian_test, canine_test, equine_test, swine_test, h1_test, h3n2_test]

avian_map = {i[2]:i[5] for i in avian_test.values}
print(len(avian_test)==len(avian_map))

canine_map = {i[2]:i[5] for i in canine_test.values}
print(len(canine_test)==len(canine_map))

equine_map = {i[2]:i[5] for i in equine_test.values}
print(len(equine_test)==len(equine_map))

swine_map = {i[2]:i[5] for i in swine_test.values}
print(len(swine_test)==len(swine_map))

h1_map = {i[2]:i[5] for i in h1_test.values}
print(len(h1_test)==len(h1_map))

h3n2_map = {i[2]:i[5] for i in h3n2_test.values}
print(len(h3n2_test)==len(h3n2_map))

aiv_map = {i[2]:i[5] for i in unique_data_all_aiv.values}
print(len(unique_data_all_aiv)==len(aiv_map))

siv_map = {i[2]:i[5] for i in unique_data_all_siv.values}
print(len(unique_data_all_siv)==len(siv_map))





## test by host


siv_seq_ls = data_all_siv['seq'].tolist()
siv_host_ls = [0 for i in range(len(siv_seq_ls))]

siv_prediction_ls = []


model.to(device)


siv_test_dataloader = DataLoader(dataset(seq=siv_seq_ls, label=siv_host_ls), batch_size=8)


for batch_seq, batch_label in  siv_test_dataloader:
    seq_ls.append(batch_seq)
    batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt")
    batch_input.to(device)
    batch_label = torch.tensor(batch_label).to(device)
    with torch.no_grad():
        outputs = model(**batch_input)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    siv_prediction_ls += predictions.clone().detach().to('cpu').tolist()
    # h1_reference_ls += batch_label.clone().detach().to('cpu').tolist()

    # metric.add_batch(predictions=predictions, references=batch_label)
# print(metric.compute())
print('siv Acc: ', accuracy_score(siv_host_ls, siv_prediction_ls))
print(Counter(siv_prediction_ls))



aiv_seq_ls = data_all_aiv['seq'].tolist()
aiv_host_ls = [0 for i in range(len(aiv_seq_ls))]

aiv_prediction_ls = []


model.to(device)


aiv_test_dataloader = DataLoader(dataset(seq=aiv_seq_ls, label=aiv_host_ls), batch_size=8)


for batch_seq, batch_label in  aiv_test_dataloader:
    seq_ls.append(batch_seq)
    batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt")
    batch_input.to(device)
    batch_label = torch.tensor(batch_label).to(device)
    with torch.no_grad():
        outputs = model(**batch_input)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    aiv_prediction_ls += predictions.clone().detach().to('cpu').tolist()
    # h1_reference_ls += batch_label.clone().detach().to('cpu').tolist()

    # metric.add_batch(predictions=predictions, references=batch_label)
# print(metric.compute())
print('aiv Acc: ', accuracy_score(aiv_host_ls, aiv_prediction_ls))
print(Counter(aiv_prediction_ls))





## test by host and segment



from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, seq, label):
        super(dataset, self).__init__()
        self.dataset_seq = seq
        self.dataset_label = label

    def __len__(self):
        return len(self.dataset_seq)
    
    def __getitem__(self, index):
        return self.dataset_seq[index], self.dataset_label[index]

res = []
for host, parts in test_data_part.items():
    res_ls = []
    if(host in ['human-AIV', 'human-SIV', 'human-seasonal-H1', 'human-seasonal-H3N2']):
        host = 'Human'
    for part_seq in parts:
        
        host_ls = [host2id_map[host] for i in range(len(part_seq))]
        print(len(part_seq))
        test_data = dataset(seq=part_seq, label=host_ls)
        
        test_dataloader = DataLoader(test_data, batch_size=64)
        prediction_ls = []
        reference_ls = []
        
        for batch_seq, batch_label in  test_dataloader:

            batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt")
            batch_input.to(device)
            batch_label = torch.tensor(batch_label).to(device)
            with torch.no_grad():
                outputs = model(**batch_input)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)   
            prediction_ls += predictions.clone().detach().to('cpu').tolist()   
            reference_ls += batch_label.clone().detach().to('cpu').tolist()   
        res_ls.append(accuracy_score(reference_ls, prediction_ls))         
        # print('Acc: ', accuracy_score(reference_ls, prediction_ls))
    res.append(res_ls)
print(res)



# test split by part

test_data_part = {'Avian':[avian_test_ha['seq'].to_list(), avian_test_mp['seq'].to_list(), 
                   avian_test_na['seq'].to_list(), avian_test_np['seq'].to_list(), 
                   avian_test_ns['seq'].to_list(), avian_test_pa['seq'].to_list(), 
                   avian_test_pb1['seq'].to_list(), avian_test_pb2['seq'].to_list()],

'Canine':[canine_test_ha['seq'].to_list(), canine_test_mp['seq'].to_list(), 
                    canine_test_na['seq'].to_list(), canine_test_np['seq'].to_list(), 
                    canine_test_ns['seq'].to_list(), canine_test_pa['seq'].to_list(),
                    canine_test_pb1['seq'].to_list(), canine_test_pb2['seq'].to_list()],

'Equine':[equine_test_ha['seq'].to_list(), equine_test_mp['seq'].to_list(), 
                    equine_test_na['seq'].to_list(), equine_test_np['seq'].to_list(), 
                    equine_test_ns['seq'].to_list(), equine_test_pa['seq'].to_list(), 
                    equine_test_pb1['seq'].to_list(), equine_test_pb2['seq'].to_list()],

# 'Other_mammals':[other_test_ha['seq'].to_list(), other_test_mp['seq'].to_list(), 
#                     other_test_na['seq'].to_list(), other_test_np['seq'].to_list(), 
#                     other_test_ns['seq'].to_list(), other_test_pa['seq'].to_list(), 
#                     other_test_pb1['seq'].to_list(), other_test_pb2['seq'].to_list()],

'Swine':[swine_test_ha['seq'].to_list(), swine_test_mp['seq'].to_list(), 
                    swine_test_na['seq'].to_list(), swine_test_np['seq'].to_list(), 
                    swine_test_ns['seq'].to_list(), swine_test_pa['seq'].to_list(), 
                    swine_test_pb1['seq'].to_list(), swine_test_pb2['seq'].to_list()], 

'human-AIV':[aiv_test_ha['seq'].to_list(), aiv_test_mp['seq'].to_list(), 
                    aiv_test_na['seq'].to_list(), aiv_test_np['seq'].to_list(), 
                    aiv_test_ns['seq'].to_list(), aiv_test_pa['seq'].to_list(), 
                    aiv_test_pb1['seq'].to_list(), aiv_test_pb2['seq'].to_list()],

'human-SIV':[siv_test_ha['seq'].to_list(), siv_test_mp['seq'].to_list(), 
                    siv_test_na['seq'].to_list(), siv_test_np['seq'].to_list(), 
                    siv_test_ns['seq'].to_list(), siv_test_pa['seq'].to_list(), 
                    siv_test_pb1['seq'].to_list(), siv_test_pb2['seq'].to_list()],

 'human-seasonal-H1':[h1_test_ha['seq'].to_list(), h1_test_mp['seq'].to_list(), 
                    h1_test_na['seq'].to_list(), h1_test_np['seq'].to_list(), 
                    h1_test_ns['seq'].to_list(), h1_test_pa['seq'].to_list(), 
                    h1_test_pb1['seq'].to_list(), h1_test_pb2['seq'].to_list()],

 'human-seasonal-H3N2':[h3n2_test_ha['seq'].to_list(), h3n2_test_mp['seq'].to_list(), 
                    h3n2_test_na['seq'].to_list(), h3n2_test_np['seq'].to_list(), 
                    h3n2_test_ns['seq'].to_list(), h3n2_test_pa['seq'].to_list(), 
                    h3n2_test_pb1['seq'].to_list(), h3n2_test_pb2['seq'].to_list()]}


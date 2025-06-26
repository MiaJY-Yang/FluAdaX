#!/usr/bin/env python
# coding: utf-8


# get data
import pandas as pd

case_test_path = './test.csv' # Input your file directory

data_case = pd.read_csv(case_test_path)


# label


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


# packages


import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
#from bio_tokenizer import BioTokenizer
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


# join sequences

case_seq = ['<sep>'.join([ii.upper() for ii in i[4:]]) for i in data_case.values]
case_label = [host2id_map['Human'] if 'Human' in i else host2id_map[i] for i in data_case['host_y'].to_list()]
# case_label = [0]*len(case_seq)   # when host is unknown or not among the five supported species


case_id = data_case['ID'].to_list()



from bio_tokenizer import BioTokenizer
import torch
from transformers import MegaConfig, MegaForSequenceClassification
import time



# get tokenizer and model
tokenizer = BioTokenizer()

# load model
model = MegaForSequenceClassification.from_pretrained('./trained_FluAdaX_G/')


# dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


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
            self.dataset_seq = case_seq
            self.dataset_label = case_label
            self.dataset_id = case_id


    def __len__(self):
        return len(self.dataset_seq)
    
    def __getitem__(self, index):
        return self.dataset_seq[index], self.dataset_label[index],self.dataset_id[index]


# test data

test_data = PropertyDataset('test')
test_dataloader = DataLoader(test_data, batch_size=8)


# prediction


import torch
from torch.nn import init

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)
device = torch.device('cpu')



prediction_ls = []
reference_ls = []
seq_ls = []
logits_ls = []
id_ls = []
model.to(device)
# batch_label,
for batch_seq, batch_label, batch_id in  tqdm(test_dataloader):
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


# confidence level

from torch.nn.functional import softmax

logits_ls_norm = [softmax(torch.Tensor(i)).detach().tolist() for i in logits_ls]

data_case['logits_ls_norm'] = logits_ls_norm


# Output to new file
output_file_path = "./case_result.xlsx"
data_case.to_excel(output_file_path, index=False, sheet_name='case预测结果')

file_path =  './case_result.xlsx'
df = pd.read_excel(file_path)

df['logits_ls_norm'] = df['logits_ls_norm'].str.strip('[]')  # 去掉方括号
df[['Human_prob', 'Avian_prob', 'Swine_prob', 'Canine_prob', 'Equine_prob']] = df['logits_ls_norm'].str.split(',', expand=True).astype(float)

# predicted_host
df['predict_host'] = df[['Human_prob', 'Avian_prob', 'Swine_prob', 'Canine_prob', 'Equine_prob']].idxmax(axis=1).str.replace('_prob', '')

output_file = './case_result.xlsx'
df.to_excel(output_file, index=False)

print(f"Well done, output to {output_file}")





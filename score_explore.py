#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import safetensors
from safetensors import safe_open
import torch
score_path = './analyses/aiv/scores_ekfac/pairwise_scores.safetensors' ##replace with user's intermediate result


tensors = {}
with safe_open(score_path, framework="pt", device="cpu")as file:
    for key in file.keys():
       tensors[key] = file.get_tensor(key)


### trainning sample selection for xgboost 

import pandas as pd

train_data = pd.read_csv('./train_ns.csv')
# train = train_data[:9167]
valid_data = pd.read_csv('./valid_ns.csv')
test_data = pd.read_csv('./test_ns.csv')

sum(test_data['host']=='human')
print(sum(train_data['host']=='human'))
print(train_data)


for key, value in tensors.items():
    print(key, value.shape)


tensors['classifier.dense'].shape


plt.figure(figsize=(20, 500))
plt.imshow(torch.clamp(tensors['classifier.out_proj']+tensors['classifier.dense'], min=-300, max=300), cmap='hot', interpolation='nearest')

plt.show()



human_all = torch.mean(torch.clamp(tensors['classifier.out_proj']+tensors['classifier.dense'], min=-300, max=300)[:65], dim=0)
avian_all = torch.mean(torch.clamp(tensors['classifier.out_proj']+tensors['classifier.dense'], min=-300, max=300)[65:], dim=0)
plt.figure(figsize=(80, 10))
# plt.bar(range(9167), avian_all)
plt.bar(range(13944), human_all)


# train

value_human_pos, index_human_pos = torch.topk(human_all,500)
# value_avian, index_avian = torch.topk(avian_all,100)
value_human_neg, index_human_neg = torch.topk(human_all,200,largest=False)


index_human_pos

import numpy as np

human_imort_pos = train_data.iloc[np.array(index_human_pos), :]
human_imort_neg = train_data.iloc[np.array(index_human_neg), :]


from collections import Counter

Counter(human_imort_pos['host']).most_common()



# human_imort_neg

Counter(human_imort_neg['host']).most_common()

human_imort_pos.to_csv('top_pos_600_human.csv',index=False)



from collections import Counter

Counter(human_imort_pos['host'])
Counter(human_imort_neg['host'])



# train xgboost

data_all = human_imort_pos

mapping = {
    '-':[1,0,0,0,0],
    'A':[0,1,0,0,0],
    'T':[0,0,1,0,0],
    'C':[0,0,0,1,0],
    'G':[0,0,0,0,1]
}

host2id = {
    'human':0,
    'avian':1
}

import numpy as np
from tqdm import tqdm

def transfer(df):
    seq_ls = df['seq']
    host_ls = df['host']

    seq_oh_ls = []
    for i in tqdm(seq_ls):
        seq_oh_it = []
        for ii in i:
            seq_oh_it += mapping[ii]
        seq_oh_ls.append(seq_oh_it)

    host_ls_id = [host2id[i] for i in host_ls]

    return np.array(seq_oh_ls), np.array(host_ls_id)




train_x, train_y = transfer(data_all)





import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. data preparation
# train_x, train_y, valid_x, valid_y, test_x, test_y = # data loading

# 2. transform into DMatrix
dtrain = xgb.DMatrix(train_x, label=train_y)
# dvalid = xgb.DMatrix(valid_x, label=valid_y)
# dtest = xgb.DMatrix(test_x, label=test_y)


# weights
labels, counts = np.unique(train_y, return_counts=True)
class_weights = {label: count for label, count in zip(labels, counts)}
max_count = max(counts)
weights = [max_count / class_weights[label] for label in train_y]

print(weights)
# update weights of DMatrix
dtrain.set_weight(weights)


# 3. parameter setting
params = {
    'objective': 'binary:logistic',  
    'eval_metric': 'logloss',        
    'eta': 0.1,                      
    'max_depth': 6,                  
    'subsample': 0.8,                
    'colsample_bytree': 0.8,         
    'seed': 42                       
}

# 4. watchlist setting
watchlist = [(dtrain, 'train'), (dtrain, 'valid')]

# 5. model training
bst = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist, early_stopping_rounds=10)

 # 6. predict & evaluate
preds = bst.predict(dtest)
predictions = [1 if value > 0.5 else 0 for value in preds]
accuracy = accuracy_score(test_y, predictions)
print(f"Accuracy: {accuracy}")

 # 7. importane features outputs
plot_importance(bst)
plt.show()


importance = bst.get_score(importance_type='total_gain')

sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
important_features = [int(k[1:]) for k, v in sorted_importance]


important_features



index_n = {
    0:'-',
    1:'A',
    2:'T',
    3:'C',
    4:'G'
}

def get_pos_n(important_features):
    res = []
    for i in important_features:
        res.append((i//5 +1, index_n[i%5]))
    return res


res = get_pos_n(important_features)


res[:20]



def transfer2a(n_res):
    a_res = []
    for i in n_res:
        ap = (i[0]-1)//3+1
        if(ap not in a_res):
            a_res.append(ap)
    return a_res




a_p = transfer2a(res)



##  amino acid posistion outputs
a_p


#!/usr/bin/env python
# coding: utf-8


import pandas as pd

data_all_path = '*.csv'  # input your file name

data_all = pd.read_csv(data_all_path)


# get date distribution

date_all = data_all['date']

from datetime import datetime

date_format = "%Y-%m-%d"

date_all_new = []
for i in date_all:
    try:
        date_all_new.append(datetime.strptime(i, date_format).date() )
    except:
        # print(i)
        date_all_new.append(datetime.strptime(i, "%Y/%m/%d").date() )

# 将date列转换为datetime格式
data_all['date'] = pd.to_datetime(df['date'])


from sklearn.preprocessing import MinMaxScaler

# 对数据进行归一化处理
scaler = MinMaxScaler()
df_grouped_normalized = pd.DataFrame(scaler.fit_transform(df_grouped), index=df_grouped.index, columns=df_grouped.columns)

# 剔除2005年以前（不含2005年）的数据
data_all = data_all[data_all['date'].dt.year >= 2005]

filtered_df = data_all[~data_all['host_y'].isin(['Human', 'Avian'])]

# 按月和host_y进行分组，并统计每组的大小
df_grouped = filtered_df.groupby([filtered_df['date'].dt.to_period('M'), 'host_y']).size().unstack(fill_value=0)

# # 对数据进行归一化处理
# scaler = MinMaxScaler()
# df_grouped_normalized = pd.DataFrame(scaler.fit_transform(df_grouped), index=df_grouped.index, columns=df_grouped.columns)
df_grouped_normalized = df_grouped


import pandas as pd

data_all_path = './dataset/data_genome.csv'   # input your sequence data here

data_all = pd.read_csv(data_all_path)


data_all['date'] = pd.to_datetime(data_all['date'], format='mixed')




host_ls = ['Avian', 'Canine', 'Equine', 'Human', 'Human-avian', 'Human-swine', 'Other_mammals', 'Swine']

host4train = ['Avian', 'Canine', 'Equine', 'Human', 'Swine']

add4test = ['Human-avian','Human-swine']




# ignore other host

data_all_avian = data_all[data_all['host_y']=='Avian']
data_all_avian['real_host'] = ['Avian' for i in range(len(data_all_avian))]
unique_data_all_avian = data_all_avian.drop_duplicates(subset=['NA_seq', 'HA_seq', 'NP_seq', 'PA_seq', 'NS_seq', 'MP_seq', 'PB1_seq', 'PB2_seq'], keep='first')
# down_unique_data_all_avian = unique_data_all_avian.sample(int(len(unique_data_all_avian)/2))


data_all_canine = data_all[data_all['host_y']=='Canine']
data_all_canine['real_host'] = ['Canine' for i in range(len(data_all_canine))]
unique_data_all_canine = data_all_canine.drop_duplicates(subset=['NA_seq', 'HA_seq', 'NP_seq', 'PA_seq', 'NS_seq', 'MP_seq', 'PB1_seq', 'PB2_seq'], keep='first')


data_all_equine = data_all[data_all['host_y']=='Equine']
data_all_equine['real_host'] = ['Equine' for i in range(len(data_all_equine))]
unique_data_all_equine = data_all_equine.drop_duplicates(subset=['NA_seq', 'HA_seq', 'NP_seq', 'PA_seq', 'NS_seq', 'MP_seq', 'PB1_seq', 'PB2_seq'], keep='first')


data_all_swine = data_all[data_all['host_y']=='Swine']
data_all_swine['real_host'] = ['Swine' for i in range(len(data_all_swine))]
unique_data_all_swine = data_all_swine.drop_duplicates(subset=['NA_seq', 'HA_seq', 'NP_seq', 'PA_seq', 'NS_seq', 'MP_seq', 'PB1_seq', 'PB2_seq'], keep='first')


# for human
data_all_human = data_all[data_all['host_y']=='Human']
data_all_human['real_host'] = ['Human' for i in range(len(data_all_human))]
unique_data_all_human = data_all_human.drop_duplicates(subset=['NA_seq', 'HA_seq', 'NP_seq', 'PA_seq', 'NS_seq', 'MP_seq', 'PB1_seq', 'PB2_seq'], keep='first')



# into test set
data_all_aiv = data_all[data_all['host_y']=='Human-avian']
data_all_aiv['real_host'] = ['Human' for i in range(len(data_all_aiv))]
unique_data_all_aiv = data_all_aiv.drop_duplicates(subset=['NA_seq', 'HA_seq', 'NP_seq', 'PA_seq', 'NS_seq', 'MP_seq', 'PB1_seq', 'PB2_seq'], keep='first')


data_all_siv = data_all[data_all['host_y']=='Human-swine']
data_all_siv['real_host'] = ['Human' for i in range(len(data_all_siv))]
unique_data_all_siv = data_all_siv.drop_duplicates(subset=['NA_seq', 'HA_seq', 'NP_seq', 'PA_seq', 'NS_seq', 'MP_seq', 'PB1_seq', 'PB2_seq'], keep='first')



def split_dataframe(df):
    # 按照date列排序
    # df['date'] = pd.to_datetime(df['date'])
    sorted_df = df.sort_values(by='date')
    # sorted_df['date'] = pd.to_datetime(sorted_df['date'])
    
    sorted_df = sorted_df[sorted_df['date'].dt.year>=2005]
    # 计算各部分的数据量
    
    total_rows = len(sorted_df)
    train_size = int(total_rows * 0.8)
    valid_size = int(total_rows * 0.1)
    test_size = total_rows - train_size - valid_size
    
    
    # 切分DataFrame
    train = sorted_df[:train_size]
    valid = sorted_df[train_size:train_size + valid_size]
    test = sorted_df[-test_size:]
    
    return train, valid, test



avian_train,    avian_valid,    avian_test  = split_dataframe(unique_data_all_avian)
canine_train,   canine_valid,   canine_test = split_dataframe(unique_data_all_canine)
equine_train,   equine_valid,   equine_test = split_dataframe(unique_data_all_equine)
swine_train,    swine_valid,    swine_test  = split_dataframe(unique_data_all_swine)

# h1_train,       h1_valid,       h1_test     = split_dataframe(unique_data_all_h3n2)
# h3n2_train,     h3n2_valid,     h3n2_test   = split_dataframe(unique_data_all_h1)
human_train,    human_valid,    human_test = split_dataframe(unique_data_all_human)




train_all = pd.concat([avian_train, canine_train, equine_train, swine_train, human_train])[['ID', 'date', 'subtype', 'host_y', 'NA_seq', 'HA_seq', 'NP_seq', 'PA_seq', 'NS_seq', 'MP_seq', 'PB1_seq', 'PB2_seq']]
valid_all = pd.concat([avian_valid, canine_valid, equine_valid, swine_valid, human_valid])[['ID', 'date', 'subtype', 'host_y', 'NA_seq', 'HA_seq', 'NP_seq', 'PA_seq', 'NS_seq', 'MP_seq', 'PB1_seq', 'PB2_seq']]
test_all = pd.concat( [avian_test,  canine_test,  equine_test,  swine_test,  human_test, data_all_aiv, data_all_siv])[['ID', 'date', 'subtype', 'host_y', 'NA_seq', 'HA_seq', 'NP_seq', 'PA_seq', 'NS_seq', 'MP_seq', 'PB1_seq', 'PB2_seq']]

from collections import Counter

print(Counter(train_all['host_y']))
print(Counter(valid_all['host_y']))
print(Counter(test_all['host_y']))

train_all.to_csv('./dataset/after_2005_genome_new/train.csv',index=None)     
valid_all.to_csv('./dataset/after_2005_genome_new/valid.csv',index=None)
test_all.to_csv('./dataset/after_2005_genome_new/test.csv',index=None)



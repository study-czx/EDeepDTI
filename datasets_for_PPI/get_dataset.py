import scipy.io
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from pathlib import Path
import os

datasets = ['Human', 'H.pylori', 'S.cerevisiae']
name_dict  = {'Human':'_Protein_', 'H.pylori':'_protein_', 'S.cerevisiae':'_protein'}

def Make_path(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        os.makedirs(data_path)

def id_map(my_id):
    id_map = {"interger_id": "origin_id"}
    for i in range(len(my_id)):
        id_map[my_id.iloc[i,0]] = i
    return id_map

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio, seed):
    dataset = np.array(dataset)
    dataset = shuffle_dataset(dataset, seed)
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

for dataset in datasets:
    all_sequences = []
    name = name_dict[dataset]
    # 读取.mat文件
    P_Protein_A_data = scipy.io.loadmat(dataset + '/P'+name+'A.mat')
    # print(P_Protein_A_data)
    P_Protein_B_data = scipy.io.loadmat(dataset + '/P'+name+'B.mat')
    N_Protein_A_data = scipy.io.loadmat(dataset + '/N'+name+'A.mat')
    N_Protein_B_data = scipy.io.loadmat(dataset + '/N'+name+'B.mat')
    # print(N_Protein_A_data)
    P_Portein_A = P_Protein_A_data['P'+name+'A']
    P_Portein_B = P_Protein_B_data['P'+name+'B']
    if dataset == 'S.cerevisiae':
        N_Portein_A = N_Protein_A_data['proteinA']
        N_Portein_B = N_Protein_B_data['proteinB']
    else:
        N_Portein_A = N_Protein_A_data['N'+name+'A']
        N_Portein_B = N_Protein_B_data['N'+name+'B']
    # print(data)
    print(len(P_Portein_A), len(P_Portein_B), len(N_Portein_A), len(N_Portein_B))
    # for i in range(len(P_Portein_A)):
    #     all_sequences.append(P_Portein_A[i][0][0])
    # for i in range(len(P_Portein_B)):
    #     all_sequences.append(P_Portein_B[i][0][0])
    # for i in range(len(N_Portein_A)):
    #     all_sequences.append(N_Portein_A[i][0][0])
    # for i in range(len(N_Portein_B)):
    #     all_sequences.append(N_Portein_B[i][0][0])
    # # print(len(all_sequences))
    # unique_all_sequences = list(set(all_sequences))
    # seq_pd = pd.DataFrame(unique_all_sequences)
    # seq_pd.columns = ['sequence']
    # seq_pd.to_csv(dataset+'/seq.csv', index=False)
    seq_pd = pd.read_csv(dataset+'/seq.csv')
    # print(len(unique_all_sequences))
    print(seq_pd)
    protein_id_map = id_map(seq_pd)
    P_samples, N_samples = [], []
    for i in range(len(P_Portein_A)):
        protein_A = protein_id_map[P_Portein_A[i][0][0]]
        protein_B = protein_id_map[P_Portein_B[i][0][0]]
        P_samples.append([protein_A, protein_B])
    for i in range(len(N_Portein_A)):
        protein_A = protein_id_map[N_Portein_A[i][0][0]]
        protein_B = protein_id_map[N_Portein_B[i][0][0]]
        N_samples.append([protein_A, protein_B])
    P_samples_pd = pd.DataFrame(P_samples, columns=['protein_A', 'protein_B'])
    N_samples_pd = pd.DataFrame(N_samples, columns=['protein_A', 'protein_B'])
    P_samples_pd.to_csv(dataset + '/P.csv', index=False)
    N_samples_pd.to_csv(dataset + '/N.csv', index=False)

    N_select_samples = N_samples_pd.sample(n=len(P_samples_pd), random_state=1)
    # print(N_select_samples)+
    total_data = np.concatenate((P_samples_pd, N_select_samples), axis=0)
    labels = np.concatenate((np.ones(len(P_samples_pd)), np.zeros(len(N_select_samples))), axis=0)
    # print(protein_id_map)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    fold_index = 1

    for train_index, test_index in kf.split(total_data, labels):
        # 获取当前fold的训练集和验证集数据
        X_train, X_test = total_data[train_index], total_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        train_pos = X_train[y_train == 1]
        train_neg = X_train[y_train == 0]
        test_pos = X_test[y_test == 1]
        test_neg = X_test[y_test == 0]
        train_pos, dev_pos = split_dataset(train_pos, 0.9, seed=1)
        train_neg, dev_neg = split_dataset(train_neg, 0.9, seed=1)
        print(len(train_pos), len(dev_pos), len(test_pos))
        print(len(train_neg), len(dev_neg), len(test_neg))

        data_path = dataset + '/fold' + str(fold_index)
        Make_path(data_path)
        pd.DataFrame(train_pos).to_csv(data_path + '/train_P.csv', index=False)
        pd.DataFrame(train_neg).to_csv(data_path + '/train_N.csv', index=False)
        pd.DataFrame(dev_pos).to_csv(data_path + '/dev_P.csv', index=False)
        pd.DataFrame(dev_neg).to_csv(data_path + '/dev_N.csv', index=False)
        pd.DataFrame(test_pos).to_csv(data_path + '/test_P.csv', index=False)
        pd.DataFrame(test_neg).to_csv(data_path + '/test_N.csv', index=False)
        fold_index += 1


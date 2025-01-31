import torch
import numpy as np
import torch.nn as nn
import funcs
import pandas as pd
import data_loader
from model import DNNNet
import torch.multiprocessing as mp
from train_test import train_model, test_model, get_result
import os
import time
from sklearn.model_selection import StratifiedKFold

funcs.setup_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
dataset_base = 'datasets_DTI/datasets/'
datasets = ['DTI']
input_types = ['e']
# predict_types = ['5_fold']
# input_types = ['e']

lr = 1e-3
wd = 1e-5
b_size = 128

n_hidden = 128
num_epoches = 300

losses = nn.BCELoss()

name_map = {'EDDTI-e': 'EDeepDTI', 'EDDTI-d': 'EDeepDTI-d', 'EDDTI-s': 'EDeepDTI-s'}

n_jobs = 6

def train_worker_with_id(args):
    # 解包参数
    (m, n, drug_emb, protein_emb, drug_name, protein_name, train_loader, dev_loader, n_dr_feats, n_p_feats, model_save_path, device, dataset) = args
    # 训练函数逻辑
    n_dr_f = len(drug_emb[0])
    n_p_f = len(protein_emb[0])

    drug_feature = torch.tensor(drug_emb, dtype=torch.float32, device=device)
    protein_feature = torch.tensor(protein_emb, dtype=torch.float32, device=device)

    model = DNNNet(n_dr_f, n_p_f, n_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model_number = str(m * n_p_feats + n)
    print(f'Drug feature: {drug_name}, length: {n_dr_f}; Protein feature: {protein_name}, length: {n_p_f}; model number: {model_number}')
    # print(drug_feature.shape, protein_feature.shape)
    # Train
    train_duration, validation_duration = train_model(drug_feature, protein_feature, model, opt, losses, train_loader,
                                                      dev_loader, num_epoches, device, model_save_path, model_number, dataset)
    return train_duration, validation_duration



def main(input_type, dataset):
    print('dataset: ', dataset)
    print('lr: ', lr)
    print('wd: ', wd)
    print('batch_size: ', b_size)
    print('n_hidden: ', n_hidden)
    print('epoches: ', num_epoches)

    save_base = 'EDDTI-' + input_type
    # get id map and features
    dr_id_map, p_id_map, Drug_features, Protein_features = data_loader.Get_feature(dataset, input_type)
    n_dr_feats, n_p_feats = len(Drug_features), len(Protein_features)

    print('number of drug feature types: ', n_dr_feats)
    print('number of protein feature types: ', n_p_feats)
    print('number of base learners: ', n_dr_feats * n_p_feats)

    drug_name_list = list(Drug_features.keys())
    protein_name_list = list(Protein_features.keys())
    drug_embedding_list = list(Drug_features.values())
    protein_embedding_list = list(Protein_features.values())
    print(drug_name_list)
    print(protein_name_list)
    del Drug_features, Protein_features

    model_save_path_base = 'model_10fold/' + save_base + '/' + dataset
    funcs.Make_path(model_save_path_base)
    # start
    base_path = dataset_base + dataset

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    P = np.loadtxt(base_path + '/DTI_P.csv', dtype=str, delimiter=",", skiprows=1)
    N = np.loadtxt(base_path + '/DTI_N.csv', dtype=str, delimiter=",", skiprows=1)
    X, Y = funcs.Get_sample(P, N, dr_id_map, p_id_map)
    k = 0
    for train_index, dev_index in skf.split(X, Y):
        fold_type = 'fold' + str(k + 1)
        print('fold: ', fold_type)
        # data load path
        load_path = base_path + '/' + fold_type
        # model save path
        model_save_path = model_save_path_base + '/' + fold_type
        funcs.Make_path(model_save_path)

        train_X, dev_X = X[train_index], X[dev_index]
        train_Y, dev_Y = Y[train_index], Y[dev_index]

        # get loader
        train_loader_list = []
        dev_loader_list = []
        for i in range(n_dr_feats * n_p_feats):  # 假设有n_models个模型
            train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
            dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)
            train_loader_list.append(train_loader)
            dev_loader_list.append(dev_loader)

        args_list = []
        for m in range(n_dr_feats):
            for n in range(n_p_feats):
                model_count = m * n_p_feats + n
                args = (m, n, drug_embedding_list[m], protein_embedding_list[n], drug_name_list[m], protein_name_list[n], train_loader_list[model_count],
                dev_loader_list[model_count], n_dr_feats, n_p_feats, model_save_path, device, dataset)
                args_list.append(args)
        # 控制最大并发数
        max_concurrent_processes = n_jobs  # 设置同时运行的最大进程数
        with mp.Pool(max_concurrent_processes) as pool:
            pool.map(train_worker_with_id, args_list)
        k = k + 1

    # test
    test_drug = np.loadtxt(base_path + '/Drug_id.csv', dtype=str, delimiter=",", skiprows=1)
    test_protein = np.loadtxt(base_path + '/Protein_id.csv', dtype=str, delimiter=",", skiprows=1)
    test_X = []
    test_Y = []
    for i in range(len(test_drug)):
        for j in range(len(test_protein)):
            test_X.append([dr_id_map[test_drug[i]], p_id_map[test_protein[j]]])
            test_Y.append([0])

    test_X, test_Y = np.array(test_X), np.array(test_Y)
    test_loader = funcs.get_test_loader(test_X, test_Y, b_size=len(test_protein))

    print('start test')
    all_scores = []
    for k in range(10):
        output_scores = []
        fold_type = 'fold' + str(k + 1)
        print('fold: ', fold_type)
        model_save_path = model_save_path_base + '/' + fold_type

        for m in range(n_dr_feats):
            for n in range(n_p_feats):
                drug_emb = drug_embedding_list[m]
                protein_emb = protein_embedding_list[n]
                n_dr_f = len(drug_emb[0])
                n_p_f = len(protein_emb[0])
                drug_feature = torch.tensor(drug_emb, dtype=torch.float32, device=device)
                protein_feature = torch.tensor(protein_emb, dtype=torch.float32, device=device)
                model_number = str(m * n_p_feats + n)

                test_model = DNNNet(n_dr_f, n_p_f, n_hidden).to(device)
                test_model.load_state_dict(torch.load(model_save_path + '/model' + str(model_number) + '.pt'))

                model_test_scores = []
                print('test model: ', model_number)
                with torch.no_grad():
                    for step, (batch_x, batch_y) in enumerate(test_loader):
                        test_model.eval()
                        b_x = batch_x.long().to(device)
                        b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                        b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                        output = test_model(drug_feature, protein_feature, b_x_dr, b_x_p)
                        score = torch.squeeze(output, dim=1)
                        scores = score.cpu().detach().numpy()
                        model_test_scores.append(scores)
                if model_number == '0':
                    output_scores = np.array(model_test_scores)
                else:
                    output_scores = np.add(output_scores, np.array(model_test_scores))
        output_scores = output_scores / (n_dr_feats * n_p_feats)
        if k == 0:
            all_scores = output_scores
        else:
            all_scores = np.add(all_scores, output_scores)

    all_scores = all_scores / 10
    all_output_pandas = pd.DataFrame(all_scores)
    all_output_pandas.to_csv("case study/All_scores_10fold_" + input_type + "_20250125.csv", index=False, header=False)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    for input_type in input_types:
        for dataset in datasets:
            main(input_type, dataset)

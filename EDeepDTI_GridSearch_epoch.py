import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import funcs
import pandas as pd
import data_loader
from model import DNNNet
import torch.multiprocessing as mp
from train_test import train_model, test_model, get_result
import os
import time

funcs.setup_seed(12345)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
dataset_base = 'datasets_DTI/datasets/'
datasets = ['DTI']
# predict_types = ['5_fold']
predict_types = ['5_fold']
input_types = ['e']

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
    m, n, drug_embedding_list, protein_embedding_list, drug_name_list, protein_name_list, train_loader, dev_loader, n_dr_feats, n_p_feats, model_save_path, device, dataset = args
    # 训练函数逻辑
    n_dr_f = len(drug_embedding_list[m][0])
    n_p_f = len(protein_embedding_list[n][0])

    drug_feature = torch.tensor(drug_embedding_list[m], dtype=torch.float32, device=device)
    protein_feature = torch.tensor(protein_embedding_list[n], dtype=torch.float32, device=device)

    model = DNNNet(n_dr_f, n_p_f, n_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model_number = str(m * n_p_feats + n)
    print(f'Drug feature: {drug_name_list[m]}, length: {n_dr_f}; Protein feature: {protein_name_list[n]}, length: {n_p_f}; model number: {model_number}')
    # Train
    train_duration, validation_duration = train_model(drug_feature, protein_feature, model, opt, losses, train_loader,
                                                      dev_loader, num_epoches, device,
                                                      model_save_path, model_number, dataset)
    return train_duration, validation_duration


def test_worker_with_id(args):
    m, n, drug_embedding_list, protein_embedding_list, drug_name_list, protein_name_list, test_loader, n_dr_feats, n_p_feats, model_save_path, device, data_save_path = args
    n_dr_f = len(drug_embedding_list[m][0])
    n_p_f = len(protein_embedding_list[n][0])
    # print(f'Drug feature: {drug_name_list[m]}, length: {n_dr_f}')
    # print(f'Protein feature: {protein_name_list[n]}, length: {n_p_f}')
    drug_feature = torch.tensor(drug_embedding_list[m], dtype=torch.float32, device=device)
    protein_feature = torch.tensor(protein_embedding_list[n], dtype=torch.float32, device=device)
    model_number = str(m * n_p_feats + n)
    # print(f'Test model number: {model_number}')
    test_model(n_dr_f, n_p_f, model_save_path, model_number, test_loader, drug_feature, protein_feature, n_hidden,
               device, data_save_path)


def main(input_type, dataset, predict_type):
    print('dataset: ', dataset)
    print('predict type: ', predict_type)
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

    model_save_path_base = 'models/' + save_base + '/' + dataset + '/' + predict_type
    funcs.Make_path(model_save_path_base)
    # start
    base_path = dataset_base + dataset + '/' + predict_type
    for k in range(5):
        time1 = time.time()
        total_train_time = 0
        total_validation_time = 0
        fold_type = 'fold' + str(k + 1)
        print('fold: ', fold_type)
        # data load path
        load_path = base_path + '/' + fold_type
        # model save path
        model_save_path = model_save_path_base + '/' + fold_type
        funcs.Make_path(model_save_path)

        train_P = np.loadtxt(load_path + '/train_P.csv', dtype=str, delimiter=',', skiprows=1)
        dev_P = np.loadtxt(load_path + '/dev_P.csv', dtype=str, delimiter=',', skiprows=1)
        train_N = np.loadtxt(load_path + '/train_N.csv', dtype=str, delimiter=',', skiprows=1)
        dev_N = np.loadtxt(load_path + '/dev_N.csv', dtype=str, delimiter=',', skiprows=1)
        print('number of DTI: ', len(train_P), len(dev_P))
        print('number of Negative DTI ', len(train_N), len(dev_N))
        # trans samples to id map and get X Y
        train_X, train_Y = funcs.Get_sample(train_P, train_N, dr_id_map, p_id_map)
        dev_X, dev_Y = funcs.Get_sample(dev_P, dev_N, dr_id_map, p_id_map)
        # get loader
        train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
        dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)

        args_list = []
        for m in range(n_dr_feats):
            for n in range(n_p_feats):
                args = (
                m, n, drug_embedding_list, protein_embedding_list, drug_name_list, protein_name_list, train_loader,
                dev_loader, n_dr_feats, n_p_feats, model_save_path, device, dataset)
                args_list.append(args)
        # 控制最大并发数
        max_concurrent_processes = n_jobs  # 设置同时运行的最大进程数
        with mp.Pool(max_concurrent_processes) as pool:
            results = pool.map(train_worker_with_id, args_list)
        for train_time, validation_time in results:
            total_train_time += train_time
            total_validation_time += validation_time
        print(f"Total training time: {total_train_time / n_jobs:.2f} seconds")
        print(f"Total validation time: {total_validation_time / n_jobs:.2f} seconds")
        time2 = time.time()
        print('time: ', time2 - time1)
    print('start test')
    for k in range(5):
        fold_type = 'fold' + str(k + 1)
        print('fold: ', fold_type)

        data_save_path = 'models/' + save_base + '/' + dataset + '/' + predict_type + '/' + fold_type
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        load_path = base_path + '/' + fold_type
        model_save_path = model_save_path_base + '/' + fold_type
        test_P = np.loadtxt(load_path + '/test_P.csv', dtype=str, delimiter=',', skiprows=1)
        test_N = np.loadtxt(load_path + '/test_N.csv', dtype=str, delimiter=',', skiprows=1)
        test_X, test_Y = funcs.Get_sample(test_P, test_N, dr_id_map, p_id_map)
        test_loader = funcs.get_test_loader(test_X, test_Y, len(test_P) + len(test_N))
        args_list = []
        for m in range(n_dr_feats):
            for n in range(n_p_feats):
                args = (m, n, drug_embedding_list, protein_embedding_list, drug_name_list, protein_name_list,
                        test_loader, n_dr_feats, n_p_feats, model_save_path, device, data_save_path)
                args_list.append(args)
        # 控制最大并发数
        max_concurrent_processes = n_jobs  # 设置同时运行的最大进程数
        with mp.Pool(max_concurrent_processes) as pool:
            pool.map(test_worker_with_id, args_list)
    result_out = get_result(model_save_path_base, n_dr_feats, n_p_feats)
    save_path = 'view_baseline_results/' + name_map[save_base] + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_out.to_csv(
        'view_baseline_results/' + name_map[save_base] + '/' + dataset + '_' + predict_type + '_score.csv',
        index=False)
    print(result_out)



if __name__ == '__main__':
    mp.set_start_method('spawn')
    for input_type in input_types:
        for dataset in datasets:
            for predict_type in predict_types:
                main(input_type, dataset, predict_type)
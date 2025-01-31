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
from sklearn.preprocessing import MinMaxScaler

funcs.setup_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
dataset_base = 'datasets_DTI/datasets/'
datasets = ['CPI']
# predict_types = ['5_fold']
predict_types = ['5_fold']
input_types = ['e-all']

name_map = {'EDDTI-e': 'EDeepDTI', 'EDDTI-d': 'EDeepDTI-d', 'EDDTI-s': 'EDeepDTI-s'}

n_jobs = 6
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


lr = 1e-3
wd = 1e-5

n_hidden = 128
num_epoches = 300

save_base = 'EDDTI-all'
losses = nn.BCELoss()
path_model_folder = 'all_test_models/'

def Get_embedding(dataset):
    dir_path = 'datasets_DTI/datasets/' + dataset
    emb_feature_path_dr = dir_path + '/drug_embedding/'
    emb_feature_path_p = dir_path + '/protein_embedding/'
    # Drug embedding
    chemberta = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_emb.csv', dtype=float, delimiter=',')
    chemberta_mtr = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_emb_MTR.csv', dtype=float, delimiter=',')
    grover = np.loadtxt(emb_feature_path_dr + 'grover.csv', dtype=float, delimiter=',')
    molformer = np.loadtxt(emb_feature_path_dr + 'Molformer_emb.csv', dtype=float, delimiter=',')
    # molclr = np.loadtxt(emb_feature_path_dr + 'molclr_emb.csv', dtype=float, delimiter=',')
    kpgt = np.loadtxt(emb_feature_path_dr + 'kpgt_emb.csv', dtype=float, delimiter=',')
    # selformer = np.loadtxt(emb_feature_path_dr + 'SELFormer_emb.csv', dtype=float, delimiter=',')

    chemberta_max = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_emb_max.csv', dtype=float, delimiter=',')
    chemberta_mtr_max = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_emb_MTR_max.csv', dtype=float, delimiter=',')
    grover_max = np.loadtxt(emb_feature_path_dr + 'grover_max.csv', dtype=float, delimiter=',')
    molformer_max = np.loadtxt(emb_feature_path_dr + 'Molformer_emb_max.csv', dtype=float, delimiter=',')
    # molclr_max = np.loadtxt(emb_feature_path_dr + 'molclr_emb_max.csv', dtype=float, delimiter=',')
    kpgt_max = np.loadtxt(emb_feature_path_dr + 'kpgt_emb_max.csv', dtype=float, delimiter=',')
    # selformer_max = np.loadtxt(emb_feature_path_dr + 'SELFormer_emb_max.csv', dtype=float, delimiter=',')

    Dr_embedding = {'chemberta': chemberta, 'chemberta_mtr':chemberta_mtr, 'grover': grover, 'molformer': molformer,
                    'kpgt': kpgt, 'chemberta_max': chemberta_max, 'chemberta_mtr_max': chemberta_mtr_max,
                    'grover_max': grover_max, 'molformer_max': molformer_max, 'kpgt_max': kpgt_max}
    # Protein embedding
    esm2 = np.loadtxt(emb_feature_path_p + 'ESM2_emb.csv', dtype=float, delimiter=',')
    protein_bert = np.loadtxt(emb_feature_path_p + 'Protein_bert_emb.csv', dtype=float, delimiter=',')
    prottrans = np.loadtxt(emb_feature_path_p + 'ProtTrans_emb.csv', dtype=float, delimiter=',')
    # tape = np.loadtxt(emb_feature_path_p + 'TAPE_emb.csv', dtype=float, delimiter=',')
    # ankh = np.loadtxt(emb_feature_path_p + 'Ankh_emb.csv', dtype=float, delimiter=',')

    esm2_max = np.loadtxt(emb_feature_path_p + 'ESM2_emb_max.csv', dtype=float, delimiter=',')
    protein_bert_max = np.loadtxt(emb_feature_path_p + 'Protein_bert_emb_max.csv', dtype=float, delimiter=',')
    prottrans_max = np.loadtxt(emb_feature_path_p + 'ProtTrans_emb_max.csv', dtype=float, delimiter=',')
    # tape_max = np.loadtxt(emb_feature_path_p + 'TAPE_emb_max.csv', dtype=float, delimiter=',')
    # ankh_max = np.loadtxt(emb_feature_path_p + 'Ankh_emb_max.csv', dtype=float, delimiter=',')

    P_embedding = {'esm2': esm2, 'protein_bert': protein_bert, 'prottrans': prottrans,
                   'esm2_max': esm2_max, 'protein_bert_max': protein_bert_max, 'prottrans_max': prottrans_max}

    return Dr_embedding, P_embedding


def Get_descriptor(dataset):
    dir_path = 'datasets_DTI/datasets/' + dataset
    feature_path_dr = dir_path + '/drug_finger/'
    feature_path_p = dir_path + '/protein_descriptor/'
    MACCS = np.loadtxt(feature_path_dr + 'MACCS.csv', dtype=float, delimiter=',', skiprows=1)
    Pubchem = np.loadtxt(feature_path_dr + 'PubChem.csv', dtype=float, delimiter=',', skiprows=1)
    ECFP4 = np.loadtxt(feature_path_dr + 'ECFP4.csv', dtype=float, delimiter=',', skiprows=1)
    FCFP4 = np.loadtxt(feature_path_dr + 'FCFP4.csv', dtype=float, delimiter=',', skiprows=1)
    Dr_finger = {'maccs': MACCS, 'pubchem': Pubchem, 'ecfp4': ECFP4, 'fcfp4': FCFP4}

    scaler = MinMaxScaler()

    TPC = np.loadtxt(feature_path_p + 'TPC.csv', dtype=float, delimiter=',')
    PAAC = np.loadtxt(feature_path_p + 'PAAC.csv', dtype=float, delimiter=',')
    PAAC = scaler.fit_transform(PAAC)

    KSCTriad = np.loadtxt(feature_path_p + 'KSCTriad.csv', dtype=float, delimiter=',')
    CKSAAP = np.loadtxt(feature_path_p + 'CKSAAP.csv', dtype=float, delimiter=',')

    CTDC = np.loadtxt(feature_path_p + 'CTDC.csv', dtype=float, delimiter=',')
    CTDT = np.loadtxt(feature_path_p + 'CTDT.csv', dtype=float, delimiter=',')
    CTDD = np.loadtxt(feature_path_p + 'CTDD.csv', dtype=float, delimiter=',')
    CTDD = scaler.fit_transform(CTDD)

    CTD = np.concatenate((CTDC, CTDT, CTDD), axis=1)
    P_seq = {'PAAC': PAAC, 'KSCTriad': KSCTriad, 'TPC': TPC, 'CKSAAP': CKSAAP, 'CTD': CTD}
    return Dr_finger, P_seq


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
    if dataset == 'CPI':
        b_size = 256
    else:
        b_size = 128

    print('dataset: ', dataset)
    print('predict type: ', predict_type)
    print('lr: ', lr)
    print('wd: ', wd)
    print('batch_size: ', b_size)
    print('n_hidden: ', n_hidden)
    print('epoches: ', num_epoches)

    save_base = 'EDDTI-' + input_type
    # get id map and features
    Drug_id, Protein_id = data_loader.Get_id(dataset)
    n_drugs, n_proteins = len(Drug_id), len(Protein_id)
    dr_id_map, p_id_map = funcs.id_map(Drug_id), funcs.id_map(Protein_id)
    Drug_features, Protein_features = Get_embedding(dataset)
    Dr_descriptor, P_descriptor = Get_descriptor(dataset)
    Drug_features.update(Dr_descriptor)
    Protein_features.update(P_descriptor)
    n_dr_feats, n_p_feats = len(Drug_features), len(Protein_features)
    print('number of drug feature types: ', n_dr_feats)
    print('number of protein feature types: ', n_p_feats)
    print('number of base learners: ', n_dr_feats * n_p_feats)

    if dataset == 'DTI':
        drug_name_list = list(Drug_features.keys())
        protein_name_list = list(Protein_features.keys())
        drug_embedding_list = list(Drug_features.values())
        protein_embedding_list = list(Protein_features.values())
        print(drug_name_list)
        print(protein_name_list)
        del Drug_features, Protein_features

        model_save_path_base = path_model_folder + save_base + '/' + dataset + '/' + predict_type
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

            data_save_path = path_model_folder + save_base + '/' + dataset + '/' + predict_type + '/' + fold_type
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
        # result_out = get_result(model_save_path_base, n_dr_feats, n_p_feats)
        #
        # save_path = 'view_baseline_results/' + name_map[save_base] + '/'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # result_out.to_csv('view_baseline_results/' + name_map[save_base] + '/' + dataset + '_' + predict_type + '_score.csv',
        #     index=False)
        # print(result_out)

    elif dataset == 'CPI':
        train_drug_id = pd.read_csv('datasets_DTI/datasets/CPI/compound_id.csv', sep=',', dtype=str)
        train_protein_id = pd.read_csv('datasets_DTI/datasets/CPI/protein_id.csv', sep=',', dtype=str)
        train_drug_id, train_protein_id = train_drug_id.iloc[:, 0].tolist(), train_protein_id.iloc[:, 0].tolist()
        dr_id_map_train, p_id_map_train = funcs.id_map(train_drug_id), funcs.id_map(train_protein_id)
        train_drug_number, train_protein_number = [dr_id_map[dr_id] for dr_id in train_drug_id], [p_id_map[p_id] for
                                                                                                  p_id in
                                                                                                  train_protein_id]


        # train
        drug_name_list = list(Drug_features.keys())
        protein_name_list = list(Protein_features.keys())
        drug_embedding_list = list(Drug_features.values())
        protein_embedding_list = list(Protein_features.values())
        train_drug_embedding_list, train_protein_embedding_list = [], []

        # print(train_drug_number, train_protein_number)
        for dr_emb in drug_embedding_list:
            train_drug_embedding_list.append(dr_emb[train_drug_number, :])
        for p_emb in protein_embedding_list:
            train_protein_embedding_list.append(p_emb[train_protein_number, :])

        del drug_embedding_list, protein_embedding_list
        del Drug_features, Protein_features

        print(drug_name_list)
        print(protein_name_list)


        model_save_path_base = 'models/' + save_base + '/' + dataset + '/' + predict_type
        funcs.Make_path(model_save_path_base)
        # start
        base_path = dataset_base + dataset + '/' + predict_type
        # for k in range(5):
        #     time1 = time.time()
        #     total_train_time = 0
        #     total_validation_time = 0
        #     fold_type = 'fold' + str(k + 1)
        #     print('fold: ', fold_type)
        #     # data load path
        #     load_path = base_path + '/' + fold_type
        #     # model save path
        #     model_save_path = model_save_path_base + '/' + fold_type
        #     funcs.Make_path(model_save_path)
        #
        #     train_P = np.loadtxt(load_path + '/train_P.csv', dtype=str, delimiter=',', skiprows=1)
        #     dev_P = np.loadtxt(load_path + '/dev_P.csv', dtype=str, delimiter=',', skiprows=1)
        #     train_N = np.loadtxt(load_path + '/train_N.csv', dtype=str, delimiter=',', skiprows=1)
        #     dev_N = np.loadtxt(load_path + '/dev_N.csv', dtype=str, delimiter=',', skiprows=1)
        #     print('number of DTI: ', len(train_P), len(dev_P))
        #     print('number of Negative DTI ', len(train_N), len(dev_N))
        #     # trans samples to id map and get X Y
        #     train_X, train_Y = funcs.Get_sample(train_P, train_N, dr_id_map_train, p_id_map_train)
        #     dev_X, dev_Y = funcs.Get_sample(dev_P, dev_N, dr_id_map_train, p_id_map_train)
        #     # get loader
        #     train_loader_list = []
        #     dev_loader_list = []
        #     for i in range(n_dr_feats * n_p_feats):  # 假设有n_models个模型
        #         train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
        #         dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)
        #         train_loader_list.append(train_loader)
        #         dev_loader_list.append(dev_loader)
        #
        #     args_list = []
        #     for m in range(n_dr_feats):
        #         for n in range(n_p_feats):
        #             model_count = m * n_p_feats + n
        #             args = (m, n, train_drug_embedding_list, train_protein_embedding_list, drug_name_list, protein_name_list,
        #                     train_loader_list[model_count], dev_loader_list[model_count], n_dr_feats, n_p_feats,
        #                     model_save_path, device, dataset)
        #             args_list.append(args)
        #     # 控制最大并发数
        #     max_concurrent_processes = n_jobs  # 设置同时运行的最大进程数
        #     with mp.Pool(max_concurrent_processes) as pool:
        #         results = pool.map(train_worker_with_id, args_list)
        #     for train_time, validation_time in results:
        #         total_train_time += train_time
        #         total_validation_time += validation_time
        #     print(f"Total training time: {total_train_time / n_jobs:.2f} seconds")
        #     print(f"Total validation time: {total_validation_time / n_jobs:.2f} seconds")
        #     time2 = time.time()
        #     print('time: ', time2 - time1)
        #
        # # test
        data_save_path_base = model_save_path_base

        # del train_drug_embedding_list, train_protein_embedding_list

        for k in range(5):
            fold_type = 'fold' + str(k + 1)
            print('fold: ', fold_type)
            data_save_path = data_save_path_base + '/' + fold_type
            if not os.path.exists(data_save_path):
                os.makedirs(data_save_path)

            load_path = base_path + '/' + fold_type
            model_save_path = model_save_path_base + '/' + fold_type

            test_P = np.loadtxt(load_path + '/test_P.csv', dtype=str, delimiter=',', skiprows=1)
            test_N = np.loadtxt(load_path + '/test_N.csv', dtype=str, delimiter=',', skiprows=1)
            test_X, test_Y = funcs.Get_sample(test_P, test_N, dr_id_map_train, p_id_map_train)
            test_loader = funcs.get_test_loader(test_X, test_Y, 2560)
            for m in range(n_dr_feats):
                for n in range(n_p_feats):
                    args = (
                    m, n, train_drug_embedding_list, train_protein_embedding_list, drug_name_list, protein_name_list,
                    test_loader, n_dr_feats, n_p_feats, model_save_path, device, data_save_path)
                    test_worker_with_id(args)
        # result_out = get_result(data_save_path_base, n_dr_feats, n_p_feats)
        # save_path = 'view_baseline_results/' + name_map[save_base] + '/'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # result_out.to_csv(
        #     'view_baseline_results/' + name_map[save_base] + '/' + dataset + '_' + predict_type + '_score.csv',
        #     index=False)
        # print(result_out)



if __name__ == '__main__':
    mp.set_start_method('spawn')
    for input_type in input_types:
        for dataset in datasets:
            for predict_type in predict_types:
                main(input_type, dataset, predict_type)


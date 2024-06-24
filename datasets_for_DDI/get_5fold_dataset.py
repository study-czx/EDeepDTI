import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import os


def Get_XY_dataset(P, N):
    P, N = np.array(P), np.array(N)
    P_list, N_list = [], []
    P_label, N_label = [], []
    for i in range(len(P)):
        P_list.append([P[i][0], P[i][1]])
        P_label.append(1)
    for j in range(len(N)):
        N_list.append([N[j][0], N[j][1]])
        N_label.append(0)
    X = np.concatenate((P_list, N_list))
    Y = np.concatenate((P_label, N_label))
    return X, Y


def trans_P_N(X_data, Y_data):
    P_data = []
    N_data = []
    for i in range(len(X_data)):
        if Y_data[i] == 1:
            P_data.append(X_data[i])
        elif Y_data[i] == 0:
            N_data.append(X_data[i])
    return P_data, N_data


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


def Make_path(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        os.makedirs(data_path)


def write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type):
    Make_path(output_path + fold_type)
    # print(len(Train_P), len(Dev_P), len(Test_P))
    # print(len(Train_N), len(Dev_N), len(Test_N))
    Train_P, Dev_P, Test_P = pd.DataFrame(Train_P), pd.DataFrame(Dev_P), pd.DataFrame(Test_P)
    Train_N, Dev_N, Test_N = pd.DataFrame(Train_N), pd.DataFrame(Dev_N), pd.DataFrame(Test_N)

    Train_P.columns, Dev_P.columns, Test_P.columns = ['drugbank_id_1', 'drugbank_id_2'], ['drugbank_id_1',
                                                     'drugbank_id_2'], ['drugbank_id_1', 'drugbank_id_2']
    Train_N.columns, Dev_N.columns, Test_N.columns = ['drugbank_id_1', 'drugbank_id_2'], ['drugbank_id_1',
                                                                                          'drugbank_id_2'], [
                                                         'drugbank_id_1', 'drugbank_id_2']
    Train_P['label'], Dev_P['label'], Test_P['label'] = 1, 1, 1
    Train_N['label'], Dev_N['label'], Test_N['label'] = 0, 0, 0
    Train, Dev, Test = pd.concat([Train_P, Train_N]), pd.concat([Dev_P, Dev_N]), pd.concat([Test_P, Test_N])
    Train.to_csv(output_path + fold_type + '/train.csv', index=False)
    Dev.to_csv(output_path + fold_type + '/dev.csv', index=False)
    Test.to_csv(output_path + fold_type + '/test.csv', index=False)


datasets = ['deep', 'miner', 'zhang']
dataset_dict = {'deep': 'DeepDDI', 'miner': 'ChChMiner', 'zhang': 'ZhangDDI'}

for dataset in datasets:
    print('dataset: ', dataset)
    load_path = dataset_dict[dataset] + '_'

    train = pd.read_csv(load_path + 'train.csv')
    dev = pd.read_csv(load_path + 'valid.csv')
    test = pd.read_csv(load_path + 'test.csv')

    train_P, train_N = train[train['label'] == 1], train[train['label'] == 0]
    dev_P, dev_N = dev[dev['label'] == 1], dev[dev['label'] == 0]
    test_P, test_N = test[test['label'] == 1], test[test['label'] == 0]
    DDI_P = pd.concat([train_P, dev_P, test_P])
    DDI_N = pd.concat([train_N, dev_N, test_N])
    print('length of P and N')
    print(len(DDI_P), len(DDI_N))

    X, Y = Get_XY_dataset(DDI_P, DDI_N)

    k_folds = 5
    Kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1)
    skf = Kfold.split(X, Y)
    output_path = dataset_dict[dataset] + '/'
    n_fold = 0
    for train_index, test_index in skf:
        fold_type = 'fold' + str(n_fold + 1)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Train_P, Train_N = trans_P_N(X_train, Y_train)
        Test_P, Test_N = trans_P_N(X_test, Y_test)

        Train_P, Dev_P = split_dataset(Train_P, 0.75, seed=1)
        Train_N, Dev_N = split_dataset(Train_N, 0.75, seed=1)
        print('length of train dev and test')
        print(len(Train_P), len(Dev_P), len(Test_P))
        print(len(Train_N), len(Dev_N), len(Test_N))
        write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type)
        n_fold = n_fold + 1

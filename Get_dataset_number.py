import numpy as np
import funcs


funcs.setup_seed(1)
dataset_base = 'datasets/datasets/'

datasets = ['DTI','CPI']
predict_types = ['5_fold', 'new_drug', 'new_protein', 'new_drug_protein']


for dataset in datasets:
    print(dataset)
    for predict_type in predict_types:
        for k in range(5):
            fold_type = 'fold' + str(k + 1)
            load_path = dataset_base + dataset + '/' + predict_type + '/' + fold_type
            train_P = np.loadtxt(load_path + '/train_P.csv', dtype=str, delimiter=',', skiprows=1)
            dev_P = np.loadtxt(load_path + '/dev_P.csv', dtype=str, delimiter=',', skiprows=1)
            test_P = np.loadtxt(load_path + '/test_P.csv', dtype=str, delimiter=',', skiprows=1)
            train_N = np.loadtxt(load_path + '/train_N.csv', dtype=str, delimiter=',', skiprows=1)
            dev_N = np.loadtxt(load_path + '/dev_N.csv', dtype=str, delimiter=',', skiprows=1)
            test_N = np.loadtxt(load_path + '/test_N.csv', dtype=str, delimiter=',', skiprows=1)
            print('number of DTI: ', len(train_P), len(dev_P), len(test_P))
            print('number of Negative DTI ', len(train_N), len(dev_N), len(test_N))




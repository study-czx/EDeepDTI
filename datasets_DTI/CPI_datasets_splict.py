import pandas as pd
import numpy as np
import random
import datasets_DTI.funcs as funcs
from sklearn.model_selection import StratifiedKFold
import networkx as nx

np.random.seed(1)
random.seed(1)


def write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type):
    funcs.Make_path(output_path + fold_type)
    print(len(Train_P), len(Dev_P), len(Test_P))
    print(len(Train_N), len(Dev_N), len(Test_N))
    Train_P, Dev_P, Test_P = pd.DataFrame(Train_P), pd.DataFrame(Dev_P), pd.DataFrame(Test_P)
    Train_N, Dev_N, Test_N = pd.DataFrame(Train_N), pd.DataFrame(Dev_N), pd.DataFrame(Test_N)
    Train_P.columns, Dev_P.columns, Test_P.columns = ['PubChem_id', 'Uniprot_id'], ['PubChem_id', 'Uniprot_id'], [
        'PubChem_id', 'Uniprot_id']
    Train_N.columns, Dev_N.columns, Test_N.columns = ['PubChem_id', 'Uniprot_id'], ['PubChem_id', 'Uniprot_id'], [
        'PubChem_id', 'Uniprot_id']
    Train_P.to_csv(output_path + fold_type + '/train_P.csv', index=False)
    Dev_P.to_csv(output_path + fold_type + '/dev_P.csv', index=False)
    Test_P.to_csv(output_path + fold_type + '/test_P.csv', index=False)
    Train_N.to_csv(output_path + fold_type + '/train_N.csv', index=False)
    Dev_N.to_csv(output_path + fold_type + '/dev_N.csv', index=False)
    Test_N.to_csv(output_path + fold_type + '/test_N.csv', index=False)


def splict_train_valid_test_CPI(type):
    print('dataset type: ', type)
    CPI_P = pd.read_csv('datasets/CPI/CPI_P.csv')
    CPI_N = pd.read_csv('datasets/CPI/CPI_N.csv')
    Compound_id = pd.read_csv('datasets/CPI/compound_id.csv')
    Protein_id = pd.read_csv('datasets/CPI/Protein_id.csv')
    Extra_CPI_P = pd.read_csv('datasets/CPI/Extra_P.csv')
    Extra_CPI_N = pd.read_csv('datasets/CPI/Extra_N.csv')
    print(len(CPI_P), len(CPI_N))
    if type == 'random':
        output_path = 'datasets/CPI/5_fold/'
        funcs.Make_path(output_path)
        X, Y = funcs.Get_XY_dataset(CPI_P, CPI_N)
        k_folds = 5
        Kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1)
        skf = Kfold.split(X, Y)
        n_fold = 0
        for train_index, test_index in skf:
            fold_type = 'fold' + str(n_fold + 1)
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            Train_P, Train_N = funcs.trans_P_N(X_train, Y_train)
            Test_P, Test_N = funcs.trans_P_N(X_test, Y_test)
            Train_P, Dev_P = funcs.split_dataset(Train_P, 0.9, seed=1)
            Train_N, Dev_N = funcs.split_dataset(Train_N, 0.9, seed=1)
            write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type)
            n_fold = n_fold + 1

    elif type == 'new_drug' or type == 'new_protein' or type == 'new_drug_protein':
        output_path = 'datasets/CPI/' + type + '/'
        funcs.Make_path(output_path)
        train_P, train_N = CPI_P, CPI_N

        for n_fold in range(5):
            fold_type = 'fold' + str(n_fold + 1)
            funcs.Make_path(output_path + fold_type)
            Train_P, Dev_P = funcs.split_dataset(train_P, 0.9, seed=n_fold)
            Train_N, Dev_N = funcs.split_dataset(train_N, 0.9, seed=n_fold)
            if type == 'new_drug':
                need_extra_CPI_P = pd.merge(Extra_CPI_P, Protein_id, how='inner', on='Uniprot_id')
                Test_P = funcs.anti_join(need_extra_CPI_P, Compound_id)
                need_extra_CPI_N = pd.merge(Extra_CPI_N, Protein_id, how='inner', on='Uniprot_id')
                Test_N = funcs.anti_join(need_extra_CPI_N, Compound_id)
                Test_P['PubChem_id'], Test_N['PubChem_id'] = Test_P['PubChem_id'].astype(int), Test_N[
                    'PubChem_id'].astype(int)
                print(len(Test_P), len(Test_N))
                write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type)

            elif type == 'new_protein':
                need_extra_CPI_P = pd.merge(Extra_CPI_P, Compound_id, how='inner', on='PubChem_id')
                Test_P = funcs.anti_join(need_extra_CPI_P, Protein_id)
                need_extra_CPI_N = pd.merge(Extra_CPI_N, Compound_id, how='inner', on='PubChem_id')
                Test_N = funcs.anti_join(need_extra_CPI_N, Protein_id)
                Test_P['PubChem_id'], Test_N['PubChem_id'] = Test_P['PubChem_id'].astype(int), Test_N[
                    'PubChem_id'].astype(int)
                print(len(Test_P), len(Test_N))
                train_P, train_N = CPI_P, CPI_N
                write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type)

            elif type == 'new_drug_protein':
                need_extra_CPI_P = funcs.anti_join(Extra_CPI_P, Compound_id)
                Test_P = funcs.anti_join(need_extra_CPI_P, Protein_id)
                need_extra_CPI_N = funcs.anti_join(Extra_CPI_N, Compound_id)
                Test_N = funcs.anti_join(need_extra_CPI_N, Protein_id)
                Test_P['PubChem_id'], Test_N['PubChem_id'] = Test_P['PubChem_id'].astype(int), Test_N[
                    'PubChem_id'].astype(int)
                print(len(Test_P), len(Test_N))
                train_P, train_N = CPI_P, CPI_N
                write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type)


types = ['random', 'new_drug', 'new_protein', 'new_drug_protein']
for type in types:
    splict_train_valid_test_CPI(type)

import pandas as pd
import numpy as np
import random
import datasets_DTI.funcs as funcs
from sklearn.model_selection import StratifiedKFold
import networkx as nx


np.random.seed(1)
random.seed(1)


def get_DTI_P_csv():
    DTI_data = pd.read_csv('processed_data/DTI_data/DTI_P_filter.csv')
    DTI_data.to_csv('datasets/DTI/DTI_P.csv', index=False)
# get_DTI_P_csv()


def get_msx_subgraph(DTI_data, Drug_id, Protein_id):
    H = nx.Graph()
    for i in range(len(Drug_id)):
        H.add_node(Drug_id.iloc[i, 0], node_type='drug')
    for j in range(len(Protein_id)):
        H.add_node(Protein_id.iloc[j, 0], node_type='protein')
    for k in range(len(DTI_data)):
        H.add_edge(DTI_data.iloc[k, 0], DTI_data.iloc[k, 1], edge_type='binds_to')
    max_H = list(H.subgraph(c) for c in nx.connected_components(H))[0]
    DTI_bench = pd.DataFrame(list(max_H.edges))
    DTI_bench.columns = ['Drugbank_id', 'Uniprot_id']
    DTI_extra = funcs.anti_join(DTI_data, DTI_bench)
    return DTI_bench, DTI_extra


def get_unrelated_pairs(drug_list, protein_list, DTI_df):
    # get all drug-protein pairs
    pairs = []
    for i in range(len(drug_list)):
        drug_id = drug_list[i]
        for j in range(len(protein_list)):
            protein_id = protein_list[j]
            pairs.append({'Drugbank_id': drug_id, 'Uniprot_id': protein_id})
    # 将列表转换为DataFrame
    All_Drug_Protein_pairs = pd.DataFrame(pairs)
    # remove known DTIs
    negative_samples_pool = funcs.anti_join(All_Drug_Protein_pairs, DTI_df)
    return negative_samples_pool


def get_extra_negative_samples(negative_pool, drug_list, protein_list):
    # select one samples for each drug and protein
    neg_pairs = []
    # select one negative samples for each drug
    for i in range(len(drug_list)):
        drug_id = drug_list[i]
        neg_this_drug = negative_pool[negative_pool['Drugbank_id'] == drug_id]
        random_number = random.sample(range(len(neg_this_drug)), 1)
        select_neg = neg_this_drug.iloc[random_number]
        neg_pairs.append({'Drugbank_id': select_neg.iloc[0, 0], 'Uniprot_id': select_neg.iloc[0, 1]})

    # select one negative samples for each protein
    for j in range(len(protein_list)):
        protein_id = protein_list[j]
        neg_this_protein = negative_pool[negative_pool['Uniprot_id'] == protein_id]
        random_number = random.sample(range(len(neg_this_protein)), 1)
        select_neg = neg_this_protein.iloc[random_number]
        neg_pairs.append({'Drugbank_id': select_neg.iloc[0, 0], 'Uniprot_id': select_neg.iloc[0, 1]})
    Select_neg_samples = pd.DataFrame(neg_pairs)
    Select_neg_samples = Select_neg_samples.drop_duplicates().reset_index(drop=True)
    return Select_neg_samples


def get_DTI_P_N():
    DTI_data = pd.read_csv('datasets/DTI/DTI_P.csv')
    Drug_id = pd.read_csv('datasets/DTI/Drug_id.csv')
    Protein_id = pd.read_csv('datasets/DTI/Protein_id.csv')
    DTI_bench, DTI_extra = get_msx_subgraph(DTI_data, Drug_id, Protein_id)
    print(len(DTI_bench), len(DTI_extra))
    DTI_bench_drug, DTI_bench_protein = sorted(set(DTI_bench['Drugbank_id'])), sorted(set(DTI_bench['Uniprot_id']))
    DTI_extra_drug, DTI_extra_protein = sorted(set(DTI_extra['Drugbank_id'])), sorted(set(DTI_extra['Uniprot_id']))
    print(len(DTI_bench_drug), len(DTI_bench_protein))
    print(len(DTI_extra_drug), len(DTI_extra_protein))
    number_negatives = len(DTI_data)
    DTI_bench_unrelated_pairs = get_unrelated_pairs(DTI_bench_drug, DTI_bench_protein, DTI_bench)
    DTI_extra_unrelated_pairs = get_unrelated_pairs(DTI_extra_drug, DTI_extra_protein, DTI_extra)
    # 针对DTI extra选取负样本
    DTI_extra_neg = get_extra_negative_samples(DTI_extra_unrelated_pairs, DTI_extra_drug, DTI_extra_protein)
    print(len(DTI_extra_neg))
    remain_number = number_negatives - len(DTI_extra_neg)
    # 针对DTI bench选取负样本
    DTI_bench_neg1 = get_extra_negative_samples(DTI_bench_unrelated_pairs, DTI_bench_drug, DTI_bench_protein)
    print(len(DTI_bench_neg1))
    remain_number2 = remain_number - len(DTI_bench_neg1)

    # 针对DTI bench select negative samples 2
    # 剩余负样本池
    negative_samples_pool_other = funcs.anti_join(DTI_bench_unrelated_pairs, DTI_bench_neg1)
    select_negative_samples_number = random.sample(range(len(negative_samples_pool_other)), remain_number2)
    DTI_bench_neg2 = negative_samples_pool_other.iloc[select_negative_samples_number].reset_index(drop=True)
    print(len(DTI_bench_neg2))
    DTI_bench_neg = pd.concat([DTI_bench_neg1, DTI_bench_neg2]).drop_duplicates().reset_index(drop=True)

    All_samples = pd.concat([DTI_extra_neg, DTI_bench_neg]).drop_duplicates()
    All_samples = All_samples.sort_values(by='Drugbank_id').reset_index(drop=True)
    print(len(All_samples))
    All_samples.to_csv('datasets/DTI/DTI_N.csv', index=False)
    DTI_bench.to_csv('datasets/DTI/DTI_bench_P.csv', index=False)
    DTI_bench_neg.to_csv('datasets/DTI/DTI_bench_N.csv', index=False)
    DTI_extra.to_csv('datasets/DTI/DTI_extra_P.csv', index=False)
    DTI_extra_neg.to_csv('datasets/DTI/DTI_extra_N.csv', index=False)



def write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type):
    funcs.Make_path(output_path + fold_type)
    print(len(Train_P), len(Dev_P), len(Test_P))
    print(len(Train_N), len(Dev_N), len(Test_N))
    Train_P, Dev_P, Test_P = pd.DataFrame(Train_P), pd.DataFrame(Dev_P), pd.DataFrame(Test_P)
    Train_N, Dev_N, Test_N = pd.DataFrame(Train_N), pd.DataFrame(Dev_N), pd.DataFrame(Test_N)
    Train_P.columns, Dev_P.columns, Test_P.columns = ['Drugbank_id', 'Uniprot_id'], ['Drugbank_id', 'Uniprot_id'], [
        'Drugbank_id', 'Uniprot_id']
    Train_N.columns, Dev_N.columns, Test_N.columns = ['Drugbank_id', 'Uniprot_id'], ['Drugbank_id', 'Uniprot_id'], [
        'Drugbank_id', 'Uniprot_id']
    Train_P.to_csv(output_path + fold_type + '/train_P.csv', index=False)
    Dev_P.to_csv(output_path + fold_type + '/dev_P.csv', index=False)
    Test_P.to_csv(output_path + fold_type + '/test_P.csv', index=False)
    Train_N.to_csv(output_path + fold_type + '/train_N.csv', index=False)
    Dev_N.to_csv(output_path + fold_type + '/dev_N.csv', index=False)
    Test_N.to_csv(output_path + fold_type + '/test_N.csv', index=False)


def splict_train_valid_test_DTI(type):
    print('dataset type: ', type)
    DTI_P = pd.read_csv('datasets/DTI/DTI_P.csv')
    DTI_N = pd.read_csv('datasets/DTI/DTI_N.csv')
    Drug_id = pd.read_csv('datasets/DTI/Drug_id.csv')
    Protein_id = pd.read_csv('datasets/DTI/Protein_id.csv')
    print(len(DTI_P), len(DTI_N))
    if type == 'random':
        output_path = 'datasets/DTI/5_fold/'
        funcs.Make_path(output_path)
        X, Y = funcs.Get_XY_dataset(DTI_P, DTI_N)
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

    elif type == 'new_drug':
        output_path = 'datasets/DTI/new_drug/'
        threshold_drug = 0.2
        funcs.Make_path(output_path)
        test_drugs_count = random.sample(range(len(Drug_id)), int(len(Drug_id) * threshold_drug))
        test_drugs_id = Drug_id.iloc[test_drugs_count]
        print(len(test_drugs_id))
        Test_P = pd.merge(DTI_P, test_drugs_id, how='inner', on='Drugbank_id')
        train_P = funcs.anti_join(DTI_P, test_drugs_id)
        Test_N = pd.merge(DTI_N, test_drugs_id, how='inner', on='Drugbank_id')
        train_N = funcs.anti_join(DTI_N, test_drugs_id)
        for n_fold in range(5):
            fold_type = 'fold' + str(n_fold + 1)
            funcs.Make_path(output_path + fold_type)
            Train_P, Dev_P = funcs.split_dataset(train_P, 0.9, seed=n_fold)
            Train_N, Dev_N = funcs.split_dataset(train_N, 0.9, seed=n_fold)
            write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type)

    elif type == 'new_protein':
        output_path = 'datasets/DTI/new_protein/'
        threshold_drug = 0.2
        funcs.Make_path(output_path)
        test_proteins_count = random.sample(range(len(Protein_id)), int(len(Protein_id) * threshold_drug))
        test_proteins_id = Protein_id.iloc[test_proteins_count]
        print(len(test_proteins_id))
        Test_P = pd.merge(DTI_P, test_proteins_id, how='inner', on='Uniprot_id')
        train_P = funcs.anti_join(DTI_P, test_proteins_id)
        Test_N = pd.merge(DTI_N, test_proteins_id, how='inner', on='Uniprot_id')
        train_N = funcs.anti_join(DTI_N, test_proteins_id)
        for n_fold in range(5):
            fold_type = 'fold' + str(n_fold + 1)
            funcs.Make_path(output_path + fold_type)
            Train_P, Dev_P = funcs.split_dataset(train_P, 0.9, seed=n_fold)
            Train_N, Dev_N = funcs.split_dataset(train_N, 0.9, seed=n_fold)
            write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type)

    elif type == 'new_drug_protein':
        output_path = 'datasets/DTI/new_drug_protein/'
        funcs.Make_path(output_path)
        Test_P = pd.read_csv('datasets/DTI/DTI_extra_P.csv')
        Test_N = pd.read_csv('datasets/DTI/DTI_extra_N.csv')
        train_P = pd.read_csv('datasets/DTI/DTI_bench_P.csv')
        train_N = pd.read_csv('datasets/DTI/DTI_bench_N.csv')
        for n_fold in range(5):
            fold_type = 'fold' + str(n_fold + 1)
            funcs.Make_path(output_path + fold_type)
            Train_P, Dev_P = funcs.split_dataset(train_P, 0.9, seed=n_fold)
            Train_N, Dev_N = funcs.split_dataset(train_N, 0.9, seed=n_fold)
            write_csv_train_dev_test(Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N, output_path, fold_type)

# select negative samples for DrugBank dataset
get_DTI_P_N()
# splict dataset for SR, SD, SP and SDP
types = ['random', 'new_drug', 'new_protein', 'new_drug_protein']
for type in types:
    splict_train_valid_test_DTI(type)

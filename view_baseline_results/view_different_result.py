import os

import numpy as np
import sklearn.metrics as skm
import funcs
import data_loader
import pandas as pd



def Get_cross_metric(dataset, drug_feature_list, protein_feature_list, type, model_path_base, n_p_feats):
    auc_list, aupr_list = [], []
    for k in range(5):
        fold_type = 'fold' + str(k + 1)
        model_save_path = model_path_base + dataset + '/' + predict_type + '/' + fold_type
        all_labels = np.loadtxt(model_save_path + '/' + type + '_labels.csv', skiprows=1)
        all_output_scores = []

        for i in drug_feature_list:
            for j in protein_feature_list:
                m = i * n_p_feats + j
                this_scores = np.loadtxt(model_save_path + '/' + type + '_scores' + str(m) + '.csv', skiprows=1)
                all_output_scores.append(this_scores)
        all_output_scores = list(np.mean(np.array(all_output_scores), axis=0))
        test_auc = skm.roc_auc_score(all_labels, all_output_scores)
        test_aupr = skm.average_precision_score(all_labels, all_output_scores)
        auc_list.append(test_auc)
        aupr_list.append(test_aupr)
    print(auc_list)
    print(aupr_list)
    mean_auc = round(sum(auc_list) / len(auc_list), 4)
    mean_aupr = round(sum(aupr_list) / len(aupr_list), 4)
    return mean_auc, mean_aupr

datasets = ['DTI', 'CPI']

predict_type = '5_fold'


def Get_metric():
    drug_embeddings = {'chemberta': 0, 'chemberta_mtr': 1, 'grover': 2, 'molformer': 3, 'kpgt': 4}
    drug_descriptors = {'maccs': 5, 'pubchem': 6, 'ecfp4': 7, 'fcfp4': 8}
    protein_embeddings = {'esm2': 0, 'protein_bert': 1, 'prottrans': 2}
    protein_embeddings_max = {'esm2_max': 3, 'protein_bert_max': 4, 'prottrans_max': 5}
    for dataset in datasets:
        model_path = '../models/EDDTI-e/'
        need_drug_lists = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        need_protein_lists = [0, 1, 2, 3, 4, 5]
        n_p_feats = 6
        test_result = Get_cross_metric(dataset, need_drug_lists, need_protein_lists, 'test', model_path, n_p_feats)
        print(test_result)

def Get_metric2():
    drug_embeddings = {'chemberta': 0, 'chemberta_mtr': 1, 'grover': 2, 'molformer': 3, 'kpgt': 4}
    drug_embeddings_max = {'chemberta_max': 5, 'chemberta_mtr_max': 6, 'grover_max': 7, 'molformer_max': 8,'kpgt_max': 9}
    drug_descriptors = {'maccs': 10, 'pubchem': 11, 'ecfp4': 12, 'fcfp4': 13}
    protein_embeddings = {'esm2': 0, 'protein_bert': 1, 'prottrans': 2}
    protein_embeddings_max = {'esm2_max': 3, 'protein_bert_max': 4, 'prottrans_max': 5}
    protein_descriptors = {'PAAC': 6, 'KSCTriad': 7, 'TPC': 8, 'CKSAAP': 9, 'CTD': 10}
    for dataset in datasets:
        model_path = '../all_test_models/EDDTI-e-all/'
        need_drug_lists = [0, 1, 2, 3, 4, 10, 11, 12, 13]
        need_protein_lists = [0, 1, 2, 3, 4, 5]
        n_p_feats = 11
        test_result = Get_cross_metric(dataset, need_drug_lists, need_protein_lists, 'test', model_path, n_p_feats)
        print(test_result)

if __name__ == '__main__':
    # Get_metric()
    Get_metric2()
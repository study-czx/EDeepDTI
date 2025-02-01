import os

import numpy as np
import sklearn.metrics as skm
import funcs
import data_loader
import pandas as pd

datasets = ['DTI', 'CPI']
input_types = ['e', 'd', 's']

predict_type = '5_fold'

# save_base = save_base
name_map = {'EDDTI-e': 'EDeepDTI', 'EDDTI-d': 'EDeepDTI-d', 'EDDTI-s': 'EDeepDTI-s'}
type_map = {'5_fold': 'SR', 'new_drug': 'SD', 'new_protein': 'SP', 'new_drug_protein': 'SDP'}

metric_list = ['AUC', 'AUPR']
save_path = '1_all_base_learner_AUC_AUPR/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def Get_metric_all(input_type, dataset):
    save_base = '../models/EDDTI-' + input_type
    n_dr_feats, n_p_feats = data_loader.Get_feature_numbers(dataset, input_type)
    all_auc_scores = []
    all_aupr_scores = []
    for i in range(n_dr_feats):
        this_drug_auc_list, this_drug_aupr_list = [], []
        for j in range(n_p_feats):
            this_auc, this_aupr = 0, 0
            m = i * n_p_feats + j
            print('药物序号为{}，蛋白序号为{}'.format(i+1, j+1))
            for k in range(5):
                fold_type = 'fold' + str(k + 1)
                model_save_path = save_base + '/' + dataset + '/' + predict_type + '/' + fold_type
                this_scores = np.loadtxt(model_save_path + '/test_scores' + str(m) + '.csv', skiprows=1)
                all_labels = np.loadtxt(model_save_path + '/test_labels.csv', skiprows=1)
                this_scores = list(np.array(this_scores))

                test_auc = skm.roc_auc_score(all_labels, this_scores)
                test_aupr = skm.average_precision_score(all_labels, this_scores)

                this_auc = this_auc + test_auc
                this_aupr = this_aupr + test_aupr
            mean_auc = round(this_auc / 5,4)
            mean_aupr = round(this_aupr / 5,4)
            this_drug_auc_list.append(mean_auc)
            this_drug_aupr_list.append(mean_aupr)
        all_auc_scores.append(this_drug_auc_list)
        all_aupr_scores.append(this_drug_aupr_list)
    all_auc_np = np.array(all_auc_scores)
    all_aupr_np = np.array(all_aupr_scores)
    print(all_auc_np)
    print(all_aupr_np)
    df_all_auc=pd.DataFrame(all_auc_np)
    df_all_aupr=pd.DataFrame(all_aupr_np)
    df_all_auc.to_csv(save_path + dataset + '_' + input_type + '_all_auc.csv')
    df_all_aupr.to_csv(save_path + dataset + '_' + input_type + '_all_aupr.csv')

    max_auc_scores = np.max(all_auc_np)
    mean_auc_scores = np.mean(all_auc_np)
    max_aupr_scores = np.max(all_aupr_scores)
    mean_aupr_scores = np.mean(all_aupr_scores)
    print('最大AUC值为： ', round(max_auc_scores, 4))
    print('平均AUC值为： ', round(mean_auc_scores, 4))
    print('最大AUPR值为： ', round(max_aupr_scores, 4))
    print('平均AUPR值为： ', round(mean_aupr_scores, 4))

if __name__ == '__main__':
    for dataset in datasets:
        for input_type in input_types:
            print(dataset, input_type)
            Get_metric_all(input_type, dataset)


import os
import pandas as pd
from scipy import stats
import data_loader
import sklearn.metrics as skm
import numpy as np
import funcs

my_method_list = ['EDeepDTI', 'EDeepDTI-d', 'EDeepDTI-s']

dataset_list1 = ['DTI', 'CPI']
input_types = ['e', 'd', 's']
name_map = {'e': 'EDeepDTI', 'd': 'EDeepDTI-d', 's': 'EDeepDTI-s'}

predict_type = '5_fold'

# save_path = '1_all_base_learner_p_values/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

metric_list = ['AUC', 'AUPR']

for dataset in dataset_list1:
    for input_type in input_types:
        for metric in metric_list:
            print(dataset, name_map[input_type], metric)
            max_p = 0
            EDeepDTI_result = pd.read_csv(name_map[input_type] + '/' + dataset + '_' + predict_type + '_score.csv')
            EDeepDTI_values = EDeepDTI_result[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
            my_values = EDeepDTI_values[metric].values.tolist()

            this_result_list = []
            save_base = '../models/EDDTI-' + input_type

            n_dr_feats, n_p_feats = data_loader.Get_feature_numbers(dataset, input_type)
            p_value_list = []
            for i in range(n_dr_feats):
                for j in range(n_p_feats):
                    m = i * n_p_feats + j
                    auc_list = []
                    aupr_list = []
                    for k in range(5):
                        fold_type = 'fold' + str(k + 1)
                        model_save_path = save_base + '/' + dataset + '/' + predict_type + '/' + fold_type
                        this_scores = np.loadtxt(model_save_path + '/test_scores' + str(m) + '.csv', skiprows=1)
                        all_labels = np.loadtxt(model_save_path + '/test_labels.csv', skiprows=1)
                        this_scores = list(np.array(this_scores))

                        test_auc = skm.roc_auc_score(all_labels, this_scores)
                        test_aupr = skm.average_precision_score(all_labels, this_scores)
                        auc_list.append(test_auc)
                        aupr_list.append(test_aupr)

                    if metric == 'AUC':
                        t_stat, p_val = stats.ttest_ind(my_values, auc_list, equal_var=False)
                    elif metric == 'AUPR':
                        t_stat, p_val = stats.ttest_ind(my_values, aupr_list, equal_var=False)
                    if p_val > max_p:
                        max_p = p_val
                    # print('p value of method {} in {} is {}'.format(baseline, metric, p_val))
                    p_value_list.append(p_val)

            # result_df = pd.DataFrame(this_result_list)
            # result_df.columns = metric_list
            # result_df.index = baseline_list
            # print(result_df)
            print('max p-value is : ' + str(max_p))
            # result_df.to_csv(save_path + dataset + '_' + type_map[predict_type] + '_p_value.csv')


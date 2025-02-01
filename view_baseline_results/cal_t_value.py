import os
import pandas as pd
from scipy import stats

baseline_list = ['DeepCPI', 'DeepConv-DTI', 'MolTrans', 'TransformerCPI', 'DLM-DTI', 'HyperAttentionDTI', 'MCANet', 'FMCA-DTI',
                 'DrugBAN', 'CmhAttCPI', 'BINDTI', 'MGNDTI', 'DeepDTA', 'DeepCDA', 'GraphDTA']

my_method_list = ['EDeepDTI', 'EDeepDTI-d', 'EDeepDTI-s']

dataset_list1 = ['DTI', 'CPI']
dataset_list2 = ['Davis_5fold', 'KIBA_5fold']

predict_types = ['5_fold', 'new_drug', 'new_protein', 'new_drug_protein']

type_map = {'5_fold': 'SR', 'new_drug': 'SD', 'new_protein': 'SP', 'new_drug_protein': 'SDP'}

save_path = '1_all_p_values/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

metric_list = ['AUC', 'AUPR', 'ACC', 'MCC', 'F1']
# metric_list = ['AUC', 'AUPR']
for dataset in dataset_list1:
    print(dataset)
    for predict_type in predict_types:
        max_p = 0
        print(predict_type)
        EDeepDTI_result = pd.read_csv('EDeepDTI/' + dataset + '_' + predict_type + '_score.csv')
        EDeepDTI_values = EDeepDTI_result[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
        this_result_list = []
        for baseline in baseline_list:
            df_result = pd.read_csv(baseline + '/' + dataset + '_' + predict_type + '_score.csv')
            df_result = df_result[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
            p_value_list = []
            for metric in metric_list:
                base_values = df_result[metric].values.tolist()
                my_values = EDeepDTI_values[metric].values.tolist()
                t_stat, p_val = stats.ttest_ind(my_values, base_values, equal_var=False)
                if p_val > max_p:
                    max_p = p_val
                # print('p value of method {} in {} is {}'.format(baseline, metric, p_val))
                p_value_list.append(p_val)
            this_result_list.append(p_value_list)
        result_df = pd.DataFrame(this_result_list)
        result_df.columns = metric_list
        result_df.index = baseline_list
        print(result_df)
        print('max p-value is : ' + str(max_p))
        result_df.to_csv(save_path + dataset + '_' + type_map[predict_type] + '_p_value.csv')


for dataset in dataset_list2:
    print(dataset)
    max_p = 0
    EDeepDTI_result = pd.read_csv('EDeepDTI/' + dataset + '_score.csv')
    EDeepDTI_values = EDeepDTI_result[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
    this_result_list = []
    for baseline in baseline_list:
        # print(baseline)
        df_result = pd.read_csv(baseline + '/' + dataset + '_score.csv')
        df_result = df_result[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
        p_value_list = []
        for metric in metric_list:
            base_values = df_result[metric].values.tolist()
            my_values = EDeepDTI_values[metric].values.tolist()
            t_stat, p_val = stats.ttest_ind(my_values, base_values, equal_var=False)
            if p_val > max_p:
                max_p = p_val
            # print('p value of method {} in {} is {}'.format(baseline, metric, p_val))
            p_value_list.append(p_val)
        this_result_list.append(p_value_list)
    result_df = pd.DataFrame(this_result_list)
    result_df.columns = metric_list
    result_df.index = baseline_list
    print(result_df)
    print('max p-value is : ' + str(max_p))
    result_df.to_csv(save_path + dataset + '_' + type_map[predict_type] + '_p_value.csv')

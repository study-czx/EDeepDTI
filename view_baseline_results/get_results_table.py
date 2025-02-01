import os
import pandas as pd

baseline_list = ['DeepCPI', 'DeepConv-DTI', 'MolTrans', 'TransformerCPI', 'DLM-DTI',  'HyperAttentionDTI', 'MCANet', 'FMCA-DTI',
                 'DrugBAN', 'CmhAttCPI', 'BINDTI', 'MGNDTI', 'DeepDTA', 'DeepCDA', 'GraphDTA']

my_method_list = ['EDeepDTI', 'EDeepDTI-d', 'EDeepDTI-s']

dataset_list1 = ['DTI', 'CPI']
dataset_list2 = ['Davis', 'KIBA']

predict_types = ['5_fold', 'new_drug', 'new_protein', 'new_drug_protein']

type_map = {'5_fold': 'SR', 'new_drug': 'SD', 'new_protein': 'SP', 'new_drug_protein': 'SDP'}

save_path = '1_all_result_tables/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

metric_list = ['AUC', 'AUPR', 'ACC', 'MCC', 'F1']

# Get EDeepDTI results
for dataset in dataset_list1:
    print(dataset)
    for predict_type in predict_types:
        print(predict_type)
        this_result_list = []
        for baseline in baseline_list:
            df_result = pd.read_csv(baseline + '/' + dataset + '_' + predict_type + '_score.csv')
            df_result = df_result[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
            df_mean = df_result.mean().round(4)
            this_result_list.append(df_mean)
        for my_method in my_method_list:
            df_result = pd.read_csv(my_method + '/' + dataset + '_' + predict_type + '_score.csv')
            df_result = df_result[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
            df_mean = df_result.mean().round(4)
            this_result_list.append(df_mean)
        result_df = pd.DataFrame(this_result_list)
        result_df.columns = metric_list
        result_df.index = baseline_list + my_method_list
        print(result_df)
        result_df.to_csv(save_path + dataset + '_' + type_map[predict_type] + '_scores.csv')

for dataset in dataset_list2:
    print(dataset)
    this_result_list = []
    for baseline in baseline_list:
        df_result = pd.read_csv(baseline + '/' + dataset + '_5fold_score.csv')
        df_result = df_result[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
        df_mean = df_result.mean().round(4)
        this_result_list.append(df_mean)
    for my_method in my_method_list:
        df_result = pd.read_csv(my_method + '/' + dataset + '_5fold_score.csv')
        df_result = df_result[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
        df_mean = df_result.mean().round(4)
        this_result_list.append(df_mean)
    result_df = pd.DataFrame(this_result_list)
    result_df.columns = metric_list
    result_df.index = baseline_list + my_method_list
    print(result_df)
    result_df.to_csv(save_path + dataset + '_SR_scores.csv')

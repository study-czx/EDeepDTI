import os
import pandas as pd

baseline_list = ['DeepCPI', 'DeepConv-DTI', 'MolTrans', 'TransformerCPI','DLM-DTI', 'HyperAttentionDTI', 'MCANet', 'FMCA-DTI',
                 'DrugBAN', 'CmhAttCPI', 'BINDTI', 'MGNDTI', 'DeepDTA', 'DeepCDA', 'GraphDTA']

my_method_list = ['EDeepDTI', 'EDeepDTI-d', 'EDeepDTI-s']

dataset_list = ['DTI', 'CPI']

predict_types = ['new_drug', 'new_protein', 'new_drug_protein']

type_map = {'5_fold': 'SR', 'new_drug': 'SD', 'new_protein': 'SP', 'new_drug_protein': 'SDP'}

save_path = '1_origin_result_tables/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

metric_list = ['AUC', 'AUPR']

# Get EDeepDTI results
for dataset in dataset_list:
    print(dataset)
    for predict_type in predict_types:
        print(predict_type)
        all_result_df = pd.DataFrame()
        for baseline in baseline_list:
            df_result = pd.read_csv(baseline + '/' + dataset + '_' + predict_type + '_score.csv')
            df_result = df_result[metric_list]
            if all_result_df.empty:
                all_result_df = df_result
            else:
                all_result_df = pd.concat([all_result_df, df_result],axis=1)

        for my_method in my_method_list:
            df_result = pd.read_csv(my_method + '/' + dataset + '_' + predict_type + '_score.csv')
            df_result = df_result[metric_list]
            all_result_df = pd.concat([all_result_df, df_result], axis=1)

        print(all_result_df)
        # result_df = pd.DataFrame(this_result_list)
        # result_df.columns = metric_list
        # result_df.index = baseline_list + my_method_list
        # print(result_df)
        all_result_df.to_csv(save_path + dataset + '_' + type_map[predict_type] + '_scores.csv')
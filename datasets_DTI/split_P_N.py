import pandas as pd

def splict_ChEMBL_data(activity_data, t_p, t_n):
    equal = activity_data[activity_data['Standard Relation'] == '=']
    greater = activity_data[activity_data['Standard Relation'] == '>']
    greater_equal = activity_data[activity_data['Standard Relation'] == '>=']
    less = activity_data[activity_data['Standard Relation'] == '<']
    less_equal = activity_data[activity_data['Standard Relation'] == '<=']
    # print(len(equal), len(greater), len(greater_equal), len(less), len(less_equal))
    # 如果以t_p为阈值选取正样本，鉴于同一个样本可能有多个取值
    positive_samples1 = equal[equal['Standard Value'] < t_p]
    positive_samples2 = less[less['Standard Value'] <= t_p]
    positive_samples3 = less_equal[less_equal['Standard Value'] < t_p]
    positive_samples = pd.concat([positive_samples1, positive_samples2, positive_samples3])
    # print(len(positive_samples))
    useless_samples1 = equal[equal['Standard Value'] >= t_p]
    useless_samples2 = greater[greater['Standard Value'] >= t_p]
    useless_samples3 = greater_equal[greater_equal['Standard Value'] >= t_p]
    useless_samples = pd.concat([useless_samples1, useless_samples2, useless_samples3])
    useless_samples = useless_samples[['Molecule ChEMBL ID', 'Target ChEMBL ID']]
    # print(len(useless_samples))
    # 合并两个 DataFrame，并筛选出只在左边 DataFrame 中出现的行
    P_data = pd.merge(positive_samples, useless_samples, on=['Molecule ChEMBL ID', 'Target ChEMBL ID'], how='left',
                      indicator=True).loc[lambda x: x['_merge'] == 'left_only']
    # 删除指示列
    P_data = P_data.reset_index(drop=True).drop(columns='_merge')
    # print(P_data)
    # 选择负样本
    negative_samples1 = equal[equal['Standard Value'] > t_n]
    negative_samples2 = greater[greater['Standard Value'] >= t_n]
    negative_samples3 = greater_equal[greater_equal['Standard Value'] > t_n]
    negative_samples = pd.concat([negative_samples1, negative_samples2, negative_samples3])
    # print(len(negative_samples))
    useless_samples1 = equal[equal['Standard Value'] <= t_n]
    useless_samples2 = less[less['Standard Value'] <= t_n]
    useless_samples3 = less_equal[less_equal['Standard Value'] <= t_n]
    useless_samples = pd.concat([useless_samples1, useless_samples2, useless_samples3])
    useless_samples = useless_samples[['Molecule ChEMBL ID', 'Target ChEMBL ID']]
    # print(len(useless_samples))
    # 合并两个 DataFrame，并筛选出只在左边 DataFrame 中出现的行
    N_data = pd.merge(negative_samples, useless_samples, on=['Molecule ChEMBL ID', 'Target ChEMBL ID'], how='left',
                      indicator=True).loc[lambda x: x['_merge'] == 'left_only']
    # 删除指示列
    N_data = N_data.reset_index(drop=True).drop(columns='_merge')
    P_data, N_data = P_data.drop_duplicates(), N_data.drop_duplicates()
    # print(N_data)
    return P_data, N_data


def splict_BindingDB_data(activity_data, t_p, t_n):
    equal = activity_data[activity_data['Standard Relation'] == '=']
    greater = activity_data[activity_data['Standard Relation'] == '>']
    less = activity_data[activity_data['Standard Relation'] == '<']
    # print(len(equal), len(greater), len(less))
    # 如果以t_p为阈值选取正样本，鉴于同一个样本可能有多个取值
    positive_samples1 = equal[equal['Standard Value'] <= t_p]
    positive_samples2 = less[less['Standard Value'] <= t_p]
    positive_samples = pd.concat([positive_samples1, positive_samples2])
    # print(len(positive_samples))
    useless_samples1 = equal[equal['Standard Value'] > t_p]
    useless_samples2 = greater[greater['Standard Value'] >= t_p]
    useless_samples = pd.concat([useless_samples1, useless_samples2])
    useless_samples = useless_samples[['PubChem_id', 'Uniprot_id']]
    # print(len(useless_samples))
    # 合并两个 DataFrame，并筛选出只在左边 DataFrame 中出现的行
    P_data = pd.merge(positive_samples, useless_samples, on=['PubChem_id', 'Uniprot_id'], how='left',
                      indicator=True).loc[lambda x: x['_merge'] == 'left_only']
    # 删除指示列
    P_data = P_data.reset_index(drop=True).drop(columns='_merge')
    # print(P_data)
    # 选择负样本
    negative_samples1 = equal[equal['Standard Value'] >= t_n]
    negative_samples2 = greater[greater['Standard Value'] >= t_n]
    negative_samples = pd.concat([negative_samples1, negative_samples2])
    # print(len(negative_samples))
    useless_samples1 = equal[equal['Standard Value'] < t_n]
    useless_samples2 = less[less['Standard Value'] <= t_n]
    useless_samples = pd.concat([useless_samples1, useless_samples2])
    useless_samples = useless_samples[['PubChem_id', 'Uniprot_id']]
    # print(len(useless_samples))
    # 合并两个 DataFrame，并筛选出只在左边 DataFrame 中出现的行
    N_data = pd.merge(negative_samples, useless_samples, on=['PubChem_id', 'Uniprot_id'], how='left',
                      indicator=True).loc[lambda x: x['_merge'] == 'left_only']
    # 删除指示列
    N_data = N_data.reset_index(drop=True).drop(columns='_merge')
    # print(N_data)
    return P_data, N_data


def get_P_N(c_data, b_data, t_p, t_n):
    print('length of chembl data:', len(c_data))
    print('length of bindingdb data:', len(b_data))
    # 1788778 1952424
    ChEMBL_P_data, ChEMBL_N_data = splict_ChEMBL_data(c_data, t_p, t_n)
    BindingDB_P_data, BindingDB_N_data = splict_BindingDB_data(b_data, t_p, t_n)
    # 2550604
    return ChEMBL_P_data, ChEMBL_N_data, BindingDB_P_data, BindingDB_N_data

def splict_P_N_data(chembl_data, bindingdb_data, t_Positive, t_Negative):
    ChEMBL_P_data, ChEMBL_N_data, BindingDB_P_data, BindingDB_N_data = get_P_N(chembl_data, bindingdb_data, t_Positive,
                                                                               t_Negative)
    new_column_name = {'PubChem id': 'PubChem_id'}
    ChEMBL_P_data = ChEMBL_P_data.rename(columns=new_column_name)
    ChEMBL_N_data = ChEMBL_N_data.rename(columns=new_column_name)
    ChEMBL_P_data_new = ChEMBL_P_data[['PubChem_id', 'Uniprot_id']].drop_duplicates()
    ChEMBL_N_data_new = ChEMBL_N_data[['PubChem_id', 'Uniprot_id']].drop_duplicates()
    BindingDB_P_data_new = BindingDB_P_data[['PubChem_id', 'Uniprot_id']].drop_duplicates()
    BindingDB_N_data_new = BindingDB_N_data[['PubChem_id', 'Uniprot_id']].drop_duplicates()

    print('length of Positive samples and Negative samples from ChEMBL: ')
    print(len(ChEMBL_P_data_new))
    print(len(ChEMBL_N_data_new))

    print('length of Positive samples and Negative samples from BindingDB: ')
    print(len(BindingDB_P_data_new))
    print(len(BindingDB_N_data_new))

    CPI_P_data = pd.concat([ChEMBL_P_data_new, BindingDB_P_data_new]).drop_duplicates()
    CPI_N_data = pd.concat([ChEMBL_N_data_new, BindingDB_N_data_new]).drop_duplicates()
    print('length of Positive samples and Negative samples of CPI: ')
    print(len(CPI_P_data))
    print(len(CPI_N_data))
    return CPI_P_data, CPI_N_data

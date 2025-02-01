import pandas as pd
import warnings

warnings.filterwarnings('ignore')

datasets = ['DTI', 'CPI']
family_list = ['Enzymes', 'GPCR', 'NR', 'IC']

for dataset in datasets:
    # 获取所有蛋白的Uniprot ID
    if dataset == 'DTI':
        all_protein_ids = pd.read_csv('../datasets_DTI/datasets/' + dataset + '/protein_id.csv')
    else:
        all_protein_ids = pd.read_csv('../datasets_DTI/datasets/' + dataset + '/all_protein_id.csv')
    all_protein_ids.columns = ['Uniprot_id']

    all_protein_family_output = pd.DataFrame()
    protein_family_kegg = pd.read_csv('KEGG/' + dataset + '/protein_family.csv')
    protein_family_chembl = pd.read_csv('ChEMBL/' + dataset + '/protein_family.csv')
    protein_family_gtopdb = pd.read_csv('GtoPdb/' + dataset + '/protein_family.csv')

    for family in family_list:
        this_family_kegg = protein_family_kegg[protein_family_kegg['Category'] == family]
        this_family_chembl = protein_family_chembl[protein_family_chembl['Category'] == family]
        this_family_gtopdb = protein_family_gtopdb[protein_family_gtopdb['Category'] == family]

        this_protein_list = this_family_kegg['Uniprot_id'].tolist() + this_family_chembl['Uniprot_id'].tolist() + \
                            this_family_gtopdb['Uniprot_id'].tolist()
        this_protein_list = sorted(list(set(this_protein_list)))

        protein_family_df = pd.DataFrame(this_protein_list, columns=['Uniprot_id'])
        protein_family_df['Family'] = family
        protein_family_df['KEGG'] = 0
        protein_family_df['ChEMBL'] = 0
        protein_family_df['GtoPdb'] = 0

        for i in range(len(protein_family_df)):
            this_protein_id = protein_family_df['Uniprot_id'][i]

            if this_protein_id in this_family_kegg['Uniprot_id'].values:
                protein_family_df['KEGG'][i] = 1

            if this_protein_id in this_family_chembl['Uniprot_id'].values:
                protein_family_df['ChEMBL'][i] = 1

            if this_protein_id in this_family_gtopdb['Uniprot_id'].values:
                protein_family_df['GtoPdb'][i] = 1

        if all_protein_family_output.empty:
            all_protein_family_output = protein_family_df
        else:
            all_protein_family_output = pd.concat([all_protein_family_output, protein_family_df])

    output_family = all_protein_family_output.reset_index(drop=True)
    output_family['score'] = output_family['KEGG'] + output_family['ChEMBL'] + output_family['GtoPdb']

    # 寻找同时属于两种家族的蛋白
    uniprot_counts = output_family['Uniprot_id'].value_counts()

    # 保存仅有一条记录的蛋白
    uniprot_one = uniprot_counts[uniprot_counts == 1].index
    output_protein_family_df1 = output_family[output_family['Uniprot_id'].isin(uniprot_one)]

    # 有两条记录的蛋白
    uniprot_twice_or_more = uniprot_counts[uniprot_counts >= 2].index
    multi_family_protein = output_family[output_family['Uniprot_id'].isin(uniprot_twice_or_more)].sort_values(
        by='Uniprot_id', ascending=True)
    # print(multi_family_protein)
    multi_protein_list_filtered = multi_family_protein.copy()

    # 首先，保留score最大的蛋白
    multi_protein_list = list(set(multi_family_protein['Uniprot_id'].tolist()))
    for i in range(len(multi_protein_list)):
        this_protein_id = multi_protein_list[i]
        this_info = multi_family_protein[multi_family_protein['Uniprot_id'] == this_protein_id]
        sorted_this_info = this_info.sort_values(by='score', ascending=False)
        max_score = sorted_this_info['score'].max()
        rows_to_remove = this_info[this_info['score'] != max_score].index
        multi_protein_list_filtered = multi_protein_list_filtered.drop(index=rows_to_remove)

    # 如果同一Uniprot_id存在多行数据，按照“NR”，“IC”，“GPCR”，“Enzyme”的顺序保留一行数据
    priority = {'NR': 1, 'IC': 2, 'GPCR': 3, 'Enzymes': 4}
    multi_protein_list_filtered['Priority'] = multi_protein_list_filtered['Family'].map(priority)
    output_protein_family_df2 = multi_protein_list_filtered.loc[
        multi_protein_list_filtered.groupby('Uniprot_id')['Priority'].idxmin()]
    output_protein_family_df2.drop(['Priority'], axis=1, inplace=True)

    output_protein_family_df = pd.concat([output_protein_family_df1, output_protein_family_df2])

    # 补全不存在于这四个家族中的数据
    missing_uniprot_ids = all_protein_ids[~all_protein_ids['Uniprot_id'].isin(output_protein_family_df['Uniprot_id'])]

    missing_uniprot_ids['Family'] = 'Others'
    missing_uniprot_ids['KEGG'] = 0
    missing_uniprot_ids['ChEMBL'] = 0
    missing_uniprot_ids['GtoPdb'] = 0
    missing_uniprot_ids['score'] = 0

    all_output = pd.concat([output_protein_family_df, missing_uniprot_ids]).sort_values(by='Uniprot_id',
                                                                                        ascending=True).reset_index(
        drop=True)
    all_output.to_csv(dataset + '_family_info.csv', index=False)

    print('number of Enzymes:', len(all_output[all_output['Family'] == 'Enzymes']))
    print('number of GPCR:', len(all_output[all_output['Family'] == 'GPCR']))
    print('number of NR:', len(all_output[all_output['Family'] == 'NR']))
    print('number of IC:', len(all_output[all_output['Family'] == 'IC']))
    print('number of Others:', len(all_output[all_output['Family'] == 'Others']))

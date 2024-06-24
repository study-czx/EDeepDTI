import pandas as pd
import re

def anti_join(data1, data2):
    data_new = pd.merge(data1, data2, indicator=True, how='outer').query(
        '_merge=="left_only"').drop('_merge', axis=1).reset_index(drop=True)
    return data_new

def get_chembl_DTI():
    chembl_data = pd.read_csv('../datasets_DTI/origin_data/ChEMBL/Drug Mechanisms.tsv', sep='\t')
    need_data = chembl_data.iloc[:, [0, 10, 14]]
    need_data = need_data[need_data['Target Organism'] == 'Homo sapiens']
    need_data = need_data.drop(columns=['Target Organism'])
    need_data.columns = ['Molecule ChEMBL ID', 'Target ChEMBL ID']
    print('length of human data', len(need_data))
    need_data = need_data.sort_values(by='Molecule ChEMBL ID').reset_index(drop=True)
    return need_data


def get_chembl_drugbank_map():
    chembl_drugbank = pd.read_table('../datasets_DTI/origin_data/ChEMBL/src1src2.txt')
    chembl_drugbank.columns = ['Molecule ChEMBL ID', 'DrugBank ID']
    return chembl_drugbank


def get_drugbank_kegg_map():
    drug_links = pd.read_csv("../datasets_DTI/origin_data/Drugbank/structure links.csv")
    drugbank_kegg_map = drug_links[['DrugBank ID', 'KEGG Drug ID']]
    drugbank_kegg_map.columns = ['DrugBank_id', 'KEGG Drug ID']
    return drugbank_kegg_map



def get_kegg_uniprot_map():
    chembl_drugbank = pd.read_csv('uniprot_kegg_map.tsv', sep='\t')
    chembl_drugbank.columns = ['Uniprot_id', 'KEGG Target ID']
    return chembl_drugbank


def get_chembl_target_data():
    target_info = pd.read_csv('../datasets_DTI/origin_data/ChEMBL/ChEMBL_target.csv', sep=';')
    need_info = target_info.iloc[:, [0, 2, 4]]
    # print('raw data number', len(need_info))
    need_info1 = need_info.dropna(subset=['UniProt Accessions'])
    # print('uniprot data number', len(need_info1))
    need_info2 = need_info1[need_info1['Organism'] == 'Homo sapiens']
    # print('human uniprot data number', len(need_info2))
    output_data = need_info2.drop(columns=['Organism'])
    output_data.columns = ['Target ChEMBL ID', 'Uniprot_id']
    output_data_new = pd.DataFrame(columns=['Target ChEMBL ID', 'Uniprot_id'])
    k = 0
    for i in range(len(output_data)):
        chembl_id = output_data.iloc[i, 0]
        uniprot_ids = output_data.iloc[i, 1]
        uniprot_list = uniprot_ids.split('|')
        for j in range(len(uniprot_list)):
            output_data_new.at[k, 'Target ChEMBL ID'] = chembl_id
            output_data_new.at[k, 'Uniprot_id'] = uniprot_list[j]
            k = k + 1
    return output_data_new


def get_ChEMBL_DTI_data():
    DTI_Chembl = get_chembl_DTI()
    print(DTI_Chembl)
    # DTI_Chembl.to_csv('ChEMBL/ChEMBL_DTI.csv', index=False)

    chembl_drugbank_idmap = get_chembl_drugbank_map()
    chembl_uniprot_idmap = get_chembl_target_data()
    DTI_Chembl_1 = pd.merge(DTI_Chembl, chembl_drugbank_idmap, on='Molecule ChEMBL ID', how='inner')
    DTI_Chembl_2 = pd.merge(DTI_Chembl_1, chembl_uniprot_idmap, on='Target ChEMBL ID', how='inner')
    print(DTI_Chembl_2)
    DTI_Chembl_output = DTI_Chembl_2[['DrugBank_id', 'Uniprot_id']].sort_values(
        by='DrugBank_id').drop_duplicates().reset_index(drop=True)
    print(DTI_Chembl_output)
    DTI_Chembl_output.to_csv('ChEMBL_DTI.csv', index=False)


def get_kegg_DTI():
    KEGG_DTI = pd.read_table('KEGG_DTI.txt')
    KEGG_DTI['Drug'] = KEGG_DTI['Drug'].fillna(method='ffill')
    # print(KEGG_DTI)
    pattern = r"\[HSA:(.*?)\]"
    # 使用正则表达式提取HSA部分并存入新列
    KEGG_DTI['HSA'] = KEGG_DTI['Target'].apply(
        lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else "")
    # print(KEGG_DTI['HSA'])

    new_KEGG_DTI = pd.DataFrame()

    for i in range(len(KEGG_DTI)):
        drug_id = KEGG_DTI['Drug'][i]
        target_ids = KEGG_DTI['HSA'][i]
        if target_ids == '':
            continue
        target_id_list = target_ids.split(' ')
        # print(len(target_id_list))
        for j in range(len(target_id_list)):
            target_id = 'hsa:' + target_id_list[j]
            this_dict = {'drug': drug_id, 'protein': target_id}
            record = pd.DataFrame.from_dict(this_dict, orient='index').T
            if new_KEGG_DTI.empty:
                new_KEGG_DTI = record
            else:
                new_KEGG_DTI = pd.concat([new_KEGG_DTI, record])
    new_KEGG_DTI.columns = ['KEGG Drug ID', 'KEGG Target ID']
    # print(new_KEGG_DTI)
    return new_KEGG_DTI
    # new_KEGG_DTI.to_csv('KEGG_DTI_all.csv', index=False)


def get_KEGG_DTI_data():
    DTI_KEGG = get_kegg_DTI()
    print(DTI_KEGG)
    drugbank_kegg_idmap = get_drugbank_kegg_map()
    kegg_uniprot_idmap = get_kegg_uniprot_map()
    DTI_KEGG_1 = pd.merge(DTI_KEGG, drugbank_kegg_idmap, on='KEGG Drug ID', how='inner')
    DTI_KEGG_2 = pd.merge(DTI_KEGG_1, kegg_uniprot_idmap, on='KEGG Target ID', how='inner')
    print(DTI_KEGG_2)
    DTI_KEGG_output = DTI_KEGG_2[['DrugBank_id', 'Uniprot_id']].sort_values(by='DrugBank_id').drop_duplicates().reset_index(drop=True)
    print(DTI_KEGG_output)
    DTI_KEGG_output.to_csv('KEGG_DTI.csv', index=False)

def get_match_DTI():
    DTI_KEGG = pd.read_csv('KEGG_DTI.csv')
    DTI_ChEMBL = pd.read_csv('ChEMBL_DTI.csv')
    Drug_id = pd.read_csv('../datasets_DTI/datasets/DTI/Drug_id.csv')
    Protein_id = pd.read_csv('../datasets_DTI/datasets/DTI/Protein_id.csv')
    Drug_id.columns = ['DrugBank_id']
    # match DrugBank dataset
    DTI_KEGG1 = pd.merge(DTI_KEGG, Drug_id, on='DrugBank_id', how='inner')
    DTI_KEGG2 = pd.merge(DTI_KEGG1, Protein_id, on='Uniprot_id', how='inner')
    DTI_ChEMBL1 = pd.merge(DTI_ChEMBL, Drug_id, on='DrugBank_id', how='inner')
    DTI_ChEMBL2 = pd.merge(DTI_ChEMBL1, Protein_id, on='Uniprot_id', how='inner')
    print(DTI_KEGG2)
    print(DTI_ChEMBL2)
    # remove P and N samples of DrugBank dataset
    DTI_P = pd.read_csv('../datasets_DTI/datasets/DTI/DTI_P.csv')
    DTI_N = pd.read_csv('../datasets_DTI/datasets/DTI/DTI_N.csv')
    DTI_P.columns = ['DrugBank_id', 'Uniprot_id']
    DTI_N.columns = ['DrugBank_id', 'Uniprot_id']
    DTI_KEGG_output1 = anti_join(DTI_KEGG2, DTI_P)
    DTI_KEGG_output2 = anti_join(DTI_KEGG_output1, DTI_N)
    DTI_ChEMBL_output1 = anti_join(DTI_ChEMBL2, DTI_P)
    DTI_ChEMBL_output2 = anti_join(DTI_ChEMBL_output1, DTI_N)
    print(DTI_KEGG_output2)
    print(DTI_ChEMBL_output2)

    DTI_KEGG_output2.to_csv('KEGG_DTI_match.csv', index=False)
    DTI_ChEMBL_output2.to_csv('ChEMBL_DTI_match.csv', index=False)



get_ChEMBL_DTI_data()
get_KEGG_DTI_data()
get_match_DTI()
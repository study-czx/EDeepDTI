import pandas as pd
import numpy as np
from pathlib import Path

# 获取从Uniprot下载的蛋白信息
def get_uniprot_review_human():
    uniprot_protein = pd.read_csv('origin_data/Uniprot/uniprotkb_reviewed_2024_02_22.tsv', sep='\t')
    uniprot_protein.columns = ['Uniprot_id', 'protein_name', 'Sequence']
    return uniprot_protein


# 获取可映射到Uniprot的人类靶标数据
def get_chembl_target_data():
    target_info = pd.read_csv('origin_data/ChEMBL/ChEMBL_target.csv', sep=';')
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


# ChEMBL ID map PubChem id
def get_chembl_pubchem_map():
    chembl_pubchem = pd.read_table('origin_data/ChEMBL/src1src22.txt')
    chembl_pubchem.columns = ['Molecule ChEMBL ID', 'PubChem id']
    return chembl_pubchem


# ChEMBL ID map DrugBank ID
def get_chembl_drugbank_map():
    chembl_drugbank = pd.read_table('origin_data/ChEMBL/src1src2.txt')
    chembl_drugbank.columns = ['Molecule ChEMBL ID', 'DrugBank ID']
    return chembl_drugbank


# DrugBank ID map PubChem id
def get_drugbank_pubchem_map():
    drugbank_pubchem = pd.read_table('origin_data/ChEMBL/src2src22.txt')
    drugbank_pubchem.columns = ['DrugBank ID', 'PubChem id']
    return drugbank_pubchem

# Uniprot ID map STRING id
def get_uniprot_string_map():
    uniprot_string = pd.read_table('origin_data/Uniprot/uniprot_string_map.tsv', sep='\t')
    uniprot_string.columns = ['Uniprot_id', 'STRING_id']
    return uniprot_string


def get_GO_anno():
    Uniprot_MF = pd.read_table('origin_data/QuickGO/MF.tsv', sep='\t')
    Uniprot_BP = pd.read_table('origin_data/QuickGO/BP.tsv', sep='\t')
    Uniprot_CC = pd.read_table('origin_data/QuickGO/CC.tsv', sep='\t')
    Uniprot_MF = Uniprot_MF[Uniprot_MF['GO ASPECT'] == 'F']
    Uniprot_BP = Uniprot_BP[Uniprot_BP['GO ASPECT'] == 'P']
    Uniprot_CC = Uniprot_CC[Uniprot_CC['GO ASPECT'] == 'C']
    Uniprot_MF = Uniprot_MF[['GENE PRODUCT ID', 'GO TERM']].drop_duplicates().reset_index(drop=True)
    Uniprot_BP = Uniprot_BP[['GENE PRODUCT ID', 'GO TERM']].drop_duplicates().reset_index(drop=True)
    Uniprot_CC = Uniprot_CC[['GENE PRODUCT ID', 'GO TERM']].drop_duplicates().reset_index(drop=True)
    # 仅保留三种GO信息均有的蛋白
    Uniprot_1 = Uniprot_MF['GENE PRODUCT ID'].drop_duplicates()
    Uniprot_2 = Uniprot_BP['GENE PRODUCT ID'].drop_duplicates()
    Uniprot_3 = Uniprot_CC['GENE PRODUCT ID'].drop_duplicates()
    Uniprot_1_2 = pd.merge(Uniprot_1, Uniprot_2, on='GENE PRODUCT ID', how='inner')
    Uniprot_1_2_3 = pd.merge(Uniprot_1_2, Uniprot_3, on='GENE PRODUCT ID', how='inner')
    Uniprot_MF_new = pd.merge(Uniprot_MF, Uniprot_1_2_3, on='GENE PRODUCT ID', how='inner')
    Uniprot_BP_new = pd.merge(Uniprot_BP, Uniprot_1_2_3, on='GENE PRODUCT ID', how='inner')
    Uniprot_CC_new = pd.merge(Uniprot_CC, Uniprot_1_2_3, on='GENE PRODUCT ID', how='inner')
    # print(len(Uniprot_MF_new), len(Uniprot_BP_new), len(Uniprot_CC_new))
    return Uniprot_MF_new, Uniprot_BP_new, Uniprot_CC_new, Uniprot_1_2_3


def get_PPI_data():
    PPI_human = pd.read_csv('origin_data/STRING/9606.protein.links.full.v12.0_STRING.txt', header=0, sep=' ')
    # database !=0 or experiments != 0
    PPI_with_experiment_database = PPI_human[(PPI_human['experiments'] != 0) | (PPI_human['database'] != 0)]
    # score >= 700
    PPI_humam_high = PPI_with_experiment_database[PPI_with_experiment_database['combined_score'] >= 700]
    PPI_need = PPI_humam_high[['protein1', 'protein2']].drop_duplicates().reset_index(drop=True)
    uniprot_string_map = get_uniprot_string_map()
    PPI_need.columns = ['STRING_id', 'protein2']
    PPI_need1 = pd.merge(PPI_need, uniprot_string_map, how='inner', on='STRING_id')
    PPI_need1.columns = ['protein1', 'STRING_id', 'Uniprot1']
    PPI_need2 = pd.merge(PPI_need1, uniprot_string_map, how='inner', on='STRING_id')
    PPI_need2.columns = ['protein1', 'protein2', 'Uniprot1', 'Uniprot2']
    PPI_output = PPI_need2[['Uniprot1', 'Uniprot2']].drop_duplicates()
    PPI_output = PPI_output.sort_values(by='Uniprot1', ascending=True).reset_index(drop=True)
    PPI_id1, PPI_id2 = PPI_output[['Uniprot1']], PPI_output[['Uniprot2']]
    PPI_id1.columns, PPI_id2.columns = ['Uniprot_id'], ['Uniprot_id']
    union_PPI_id = pd.concat([PPI_id1, PPI_id2]).drop_duplicates().reset_index(drop=True)
    return PPI_output, union_PPI_id


def get_DDI_data():
    DDI_data = pd.read_csv('processed_data/Drugbank_DDI_2560048.csv')
    DDI_id = DDI_data[['drugbank_id1']].drop_duplicates().reset_index(drop=True)
    return DDI_data, DDI_id



# get ctd data from origin CTD data files
def get_ctd_data():
    CTD_chemical_disease = pd.read_csv('origin_data/CTD/CTD_chemicals_diseases.csv', comment='#', low_memory=False,
                                       header=None, usecols=[1, 4, 5])
    CTD_genes_disease = pd.read_csv('origin_data/CTD/CTD_genes_diseases.csv', comment='#', low_memory=False,
                                    header=None, usecols=[1, 3, 4])
    CTD_chemical_disease_need = CTD_chemical_disease[CTD_chemical_disease[5].notna()].reset_index(drop=True)
    CTD_genes_disease_need = CTD_genes_disease[CTD_genes_disease[4].notna()].reset_index(drop=True)
    CTD_chemical_disease_need.columns = ['chemical_id', 'disease_id', 'type']
    CTD_genes_disease_need.columns = ['gene_id', 'disease_id', 'type']
    return CTD_chemical_disease_need, CTD_genes_disease_need


def get_ctd_p_d_with_uniprot():
    CTD_genes_disease_data = pd.read_csv('processed_data/CTD/CTD_p_d.csv')
    CTD_genes_disease_m = CTD_genes_disease_data[CTD_genes_disease_data['type'] == 'marker/mechanism']
    CTD_genes_disease_t = CTD_genes_disease_data[CTD_genes_disease_data['type'] == 'therapeutic']
    CTD_genes_disease_m = CTD_genes_disease_m[['gene_id', 'disease_id']].reset_index(drop=True)
    CTD_genes_disease_t = CTD_genes_disease_t[['gene_id', 'disease_id']].reset_index(drop=True)
    Uniprot_geneid_map = pd.read_csv('origin_data/Uniprot/uniprot_geneid_map.tsv', sep='\t')
    Uniprot_geneid_map.columns = ['uniprot_id', 'gene_id']
    CTD_genes_disease_m_new = pd.merge(CTD_genes_disease_m, Uniprot_geneid_map, how='inner', on='gene_id')
    CTD_genes_disease_t_new = pd.merge(CTD_genes_disease_t, Uniprot_geneid_map, how='inner', on='gene_id')
    uniprot_disease_m = CTD_genes_disease_m_new[['uniprot_id', 'disease_id']].drop_duplicates().reset_index(drop=True)
    uniprot_disease_t = CTD_genes_disease_t_new[['uniprot_id', 'disease_id']].drop_duplicates().reset_index(drop=True)
    uniprot_disease_m = uniprot_disease_m.sort_values(by='uniprot_id', ascending=True).reset_index(drop=True)
    uniprot_disease_t = uniprot_disease_t.sort_values(by='uniprot_id', ascending=True).reset_index(drop=True)

    ctd_g_d_uniprot_id = pd.concat([uniprot_disease_m, uniprot_disease_t])
    ctd_g_d_uniprot_id = ctd_g_d_uniprot_id[['uniprot_id']].drop_duplicates()
    ctd_g_d_uniprot_id = ctd_g_d_uniprot_id.sort_values(by='uniprot_id', ascending=True).reset_index(drop=True)
    return uniprot_disease_m, uniprot_disease_t, ctd_g_d_uniprot_id

def filter_oneclass_drug_proteins(CPI_P, CPI_N):
    CPI_P_new = CPI_P
    CPI_N_new = CPI_N
    for i in range(100):
        print('round ', i)
        compound1, compound2 = CPI_P_new[['PubChem_id']].drop_duplicates(), CPI_N_new[['PubChem_id']].drop_duplicates()
        compound_inner = pd.merge(compound1, compound2, how='inner').drop_duplicates()
        if len(compound1) > len(compound_inner) or len(compound2) > len(compound_inner):
            CPI_P_new = pd.merge(CPI_P_new, compound_inner, how='inner', on='PubChem_id')
            CPI_N_new = pd.merge(CPI_N_new, compound_inner, how='inner', on='PubChem_id')
            print(len(CPI_P_new), len(CPI_N_new))
            protein1, protein2 = CPI_P_new[['Uniprot_id']].drop_duplicates(), CPI_N_new[
                ['Uniprot_id']].drop_duplicates()
            protein_inner = pd.merge(protein1, protein2, how='inner').drop_duplicates()
            if len(protein1) > len(protein_inner) or len(protein2) > len(protein_inner):
                CPI_P_new = pd.merge(CPI_P_new, protein_inner, how='inner', on='Uniprot_id')
                CPI_N_new = pd.merge(CPI_N_new, protein_inner, how='inner', on='Uniprot_id')
                print(len(CPI_P_new), len(CPI_N_new))
            else:
                break
        else:
            break
    compound_inner = compound_inner.sort_values(by='PubChem_id', ascending=True).reset_index(drop=True)
    protein_inner = protein_inner.sort_values(by='Uniprot_id', ascending=True).reset_index(drop=True)
    CPI_P_new = CPI_P_new.sort_values(by='PubChem_id', ascending=True).reset_index(drop=True)
    CPI_N_new = CPI_N_new.sort_values(by='PubChem_id', ascending=True).reset_index(drop=True)
    return compound_inner, protein_inner, CPI_P_new, CPI_N_new

# 根据正样本和负样本来得到X和Y
def Get_XY_dataset(P, N):
    P, N = np.array(P), np.array(N)
    P_list, N_list = [], []
    P_label, N_label = [], []
    for i in range(len(P)):
        P_list.append([P[i][0], P[i][1]])
        P_label.append(1)
    for j in range(len(N)):
        N_list.append([N[j][0], N[j][1]])
        N_label.append(0)
    X = np.concatenate((P_list, N_list))
    Y = np.concatenate((P_label, N_label))
    return X, Y

# 打乱数据集
def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

# 按照比例划分数据集
def split_dataset(dataset, ratio, seed):
    dataset = np.array(dataset)
    dataset = shuffle_dataset(dataset, seed)
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

# 根据X,Y获取正样本和负样本
def trans_P_N(X_data, Y_data):
    P_data = []
    N_data = []
    for i in range(len(X_data)):
        if Y_data[i] == 1:
            P_data.append(X_data[i])
        elif Y_data[i] == 0:
            N_data.append(X_data[i])
    return P_data, N_data

def Make_path(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        data_path.mkdir()

def anti_join(data1, data2):
    data_new = pd.merge(data1, data2, indicator=True, how='outer').query(
        '_merge=="left_only"').drop('_merge', axis=1).reset_index(drop=True)
    return data_new
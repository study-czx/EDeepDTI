import pandas as pd
from datasets_DTI.get_CPI_activity import get_cpi_data
from datasets_DTI.get_DTI_data import get_drugbank_DTI
from datasets_DTI.split_P_N import splict_P_N_data
from datasets_DTI.get_compound_protein_info import get_compound_info, get_uniprot_info, get_drug_info
import datasets_DTI.funcs as funcs


# get all cpi activity data from chembl database and bindingdb database
def get_cpi():
    chembl_cpi, bindingdb_cpi = get_cpi_data()
    chembl_cpi.to_csv('processed_data/CPI_data/ChEMBL_activity_data.csv', index=False)
    bindingdb_cpi.to_csv('processed_data/CPI_data/BindingDB_activity_data.csv', index=False)


# get Positive samples and Negative samples according to threshold
def get_P_N():
    chembl_cpi = pd.read_csv('processed_data/CPI_data/ChEMBL_activity_data.csv')
    bindingdb_cpi = pd.read_csv('processed_data/CPI_data/BindingDB_activity_data.csv')
    t_Positive = 100
    t_Negative = 30000
    CPI_P_data, CPI_N_data = splict_P_N_data(chembl_cpi, bindingdb_cpi, t_Positive, t_Negative)
    CPI_P_data.to_csv('processed_data/CPI_data/CPI_P.csv', index=False)
    CPI_N_data.to_csv('processed_data/CPI_data/CPI_N.csv', index=False)
    return CPI_P_data, CPI_N_data

# get pubchem compound with unique CanonicalSMILES (MolecularWeight<1000)
# get uniprot protein with unique Sequence
# if fail downloading pubchem smiles, try again
def get_info():
    CPI_P_data = pd.read_csv('processed_data/CPI_data/CPI_P.csv')
    CPI_N_data = pd.read_csv('processed_data/CPI_data/CPI_N.csv')
    all_CPI_data = pd.concat([CPI_P_data, CPI_N_data])
    compound_info = get_compound_info(all_CPI_data)
    protein_info = get_uniprot_info(all_CPI_data)
    compound_info.to_csv('processed_data/cpi_compound_info.csv', index=False)
    protein_info.to_csv('processed_data/cpi_protein_info.csv', index=False)
    return compound_info, protein_info
# get_info()

def get_GO_PPI():
    Uniprot_MF, Uniprot_BP, Uniprot_CC, Uniprot_id_GO = funcs.get_GO_anno()
    PPI_data, PPI_id = funcs.get_PPI_data()
    Uniprot_MF.to_csv('processed_data/Uniprot_MF.csv', index=False)
    Uniprot_BP.to_csv('processed_data/Uniprot_BP.csv', index=False)
    Uniprot_CC.to_csv('processed_data/Uniprot_CC.csv', index=False)
    PPI_data.to_csv('processed_data/PPI.csv', index=False)
    return Uniprot_MF, Uniprot_BP, Uniprot_CC, PPI_data


# def filter_CTD():
#     CTD_chemical_disease_need, CTD_genes_disease_need = funcs.get_ctd_data()
#     CTD_chemical_disease_need.to_csv('processed_data/CTD/CTD_dr_d.csv', index=False)
#     CTD_genes_disease_need.to_csv('processed_data/CTD/CTD_p_d.csv', index=False)
#
#
# def get_uniprot_disease():
#     uniprot_disease_m, uniprot_disease_t, ctd_g_d_uniprot_id = funcs.get_ctd_p_d_with_uniprot()
#     uniprot_disease_m.to_csv('processed_data/CTD/uniprot_disease_m.csv', index=False)
#     uniprot_disease_t.to_csv('processed_data/CTD/uniprot_disease_t.csv', index=False)

# filter needed CPI dataset
def fileter_cpi():
    CPI_P = pd.read_csv('processed_data/CPI_data/CPI_P.csv')
    CPI_N = pd.read_csv('processed_data/CPI_data/CPI_N.csv')
    print(len(CPI_P), len(CPI_N))
    # filter protein
    protein_info = pd.read_csv('processed_data/cpi_protein_info.csv')
    protein_info_id = protein_info[['Uniprot_id']].reset_index(drop=True).drop_duplicates()
    _, _, _, Uniprot_id_GO = funcs.get_GO_anno()
    _, PPI_id = funcs.get_PPI_data()
    Uniprot_id_GO.columns, PPI_id.columns = ['Uniprot_id'], ['Uniprot_id']
    Uniprot_id_with_info = pd.merge(Uniprot_id_GO, PPI_id, how='inner')
    Uniprot_id_with_info = pd.merge(Uniprot_id_with_info, protein_info_id, how='inner')

    CPI_P_new = pd.merge(CPI_P, Uniprot_id_with_info, how='inner')
    CPI_N_new = pd.merge(CPI_N, Uniprot_id_with_info, how='inner')
    print(len(CPI_P_new), len(CPI_N_new))
    # filter compound
    compound_info = pd.read_csv('processed_data/cpi_compound_info.csv', index_col=False)
    # print(compound_info)
    need_compound = compound_info[['CID']].reset_index(drop=True).drop_duplicates()
    # print(need_compound)
    need_compound.columns = ['PubChem_id']

    CPI_P_new = pd.merge(CPI_P_new, need_compound, how='inner')
    CPI_N_new = pd.merge(CPI_N_new, need_compound, how='inner')
    print(len(CPI_P_new), len(CPI_N_new))

    CPI_P_new.to_csv('processed_data/CPI_data/CPI_P_filter.csv', index=False)
    CPI_N_new.to_csv('processed_data/CPI_data/CPI_N_filter.csv', index=False)
    All_compound = sorted(set(list(CPI_P_new['PubChem_id']) + list(CPI_N_new['PubChem_id'])))
    All_protein = sorted(set(list(CPI_P_new['Uniprot_id']) + list(CPI_N_new['Uniprot_id'])))
    All_compound, All_protein = pd.DataFrame(All_compound, columns=['PubChem_id']), pd.DataFrame(All_protein,
                                                                                               columns=['Uniprot_id'])
    All_compound.to_csv('datasets/CPI/all_compound_id.csv', index=False)
    All_protein.to_csv('datasets/CPI/all_protein_id.csv', index=False)
    # 获取需要用到的化合物和蛋白的结构字符串
    output_compound = compound_info[['CID', 'CanonicalSMILES']].reset_index(drop=True)
    output_compound.columns = ['PubChem_id', 'CanonicalSMILES']
    need_compound_info = pd.merge(output_compound, All_compound, how='inner')
    need_protein_info = pd.merge(protein_info, All_protein, how='inner')
    need_compound_info.to_csv('datasets/CPI/all_compound_smiles.csv', index=False)
    need_protein_info.to_csv('datasets/CPI/all_protein_sequence.csv', index=False)

# fileter_cpi()

def fileter_cpi_with_bi_compound_protein():
    CPI_P = pd.read_csv('processed_data/CPI_data/CPI_P_filter.csv')
    CPI_N = pd.read_csv('processed_data/CPI_data/CPI_N_filter.csv')
    print(len(CPI_P), len(CPI_N))
    compound_inner, protein_inner, CPI_P_new, CPI_N_new = funcs.filter_oneclass_drug_proteins(CPI_P, CPI_N)
    CPI_other_P = funcs.anti_join(CPI_P, CPI_P_new)
    CPI_other_N = funcs.anti_join(CPI_N, CPI_N_new)
    compound_inner.to_csv('datasets/CPI/compound_id.csv', index=False)
    protein_inner.to_csv('datasets/CPI/protein_id.csv', index=False)
    CPI_P_new.to_csv('datasets/CPI/CPI_P.csv', index=False)
    CPI_N_new.to_csv('datasets/CPI/CPI_N.csv', index=False)
    CPI_other_P.to_csv('datasets/CPI/Extra_P.csv', index=False)
    CPI_other_N.to_csv('datasets/CPI/Extra_N.csv', index=False)
    print(len(CPI_P_new), len(CPI_N_new))
    print(len(CPI_other_P), len(CPI_other_N))


# get known DTIs from DrugBank dataset
def get_drugbank_dti():
    DTI_data = get_drugbank_DTI()
    print(DTI_data)
    DTI_data.to_csv('processed_data/DTI_data/DTI_P.csv', index=False)

# filter DrugBank dataset with needed drugs and needed proteins
def filter_drugbank_dti():
    DTI_data = pd.read_csv('processed_data/DTI_data/DTI_P.csv')
    DTI_data.columns = ['Drugbank_id', 'Uniprot_id']
    print(len(DTI_data))
    # protein
    _, _, _, Uniprot_id_GO = funcs.get_GO_anno()
    _, PPI_id = funcs.get_PPI_data()
    uniprot_info = get_uniprot_info(DTI_data)
    uniprot_info_id = uniprot_info[['Uniprot_id']].reset_index(drop=True).drop_duplicates()
    # drug
    _, DDI_id = funcs.get_DDI_data()
    Drugbank_info = get_drug_info()
    need_Drug = Drugbank_info[['DrugBank ID']]

    Uniprot_id_GO.columns, PPI_id.columns = ['Uniprot_id'], ['Uniprot_id']
    Uniprot_id_with_info = pd.merge(Uniprot_id_GO, PPI_id, how='inner')
    Uniprot_id_with_info = pd.merge(Uniprot_id_with_info, uniprot_info_id, how='inner')

    DTI_data_new = pd.merge(DTI_data, Uniprot_id_with_info, how='inner', on='Uniprot_id')

    DDI_id.columns, need_Drug.columns = ['Drugbank_id'], ['Drugbank_id']

    Drugbank_id_with_info = pd.merge(DDI_id, need_Drug, how='inner')
    DTI_data_new = pd.merge(DTI_data_new, Drugbank_id_with_info, how='inner', on='Drugbank_id')
    DTI_data_new = DTI_data_new.sort_values(by='Drugbank_id', ascending=True).reset_index(drop=True)
    print(len(DTI_data_new))
    DTI_data_new.to_csv('processed_data/DTI_data/DTI_P_filter.csv', index=False)

    Drug_id = DTI_data_new[['Drugbank_id']].drop_duplicates()
    Protein_id = DTI_data_new[['Uniprot_id']].drop_duplicates()
    Drug_id = Drug_id.sort_values(by='Drugbank_id', ascending=True).reset_index(drop=True)
    Protein_id = Protein_id.sort_values(by='Uniprot_id', ascending=True).reset_index(drop=True)
    print(len(Drug_id), len(Protein_id))
    Drug_id.to_csv('datasets/DTI/Drug_id.csv', index=False)
    Protein_id.to_csv('datasets/DTI/Protein_id.csv', index=False)

    Drugbank_info.columns = ['Drugbank_id', 'SMILES']
    need_drug_info = pd.merge(Drugbank_info, Drug_id, how='inner')
    need_protein_info = pd.merge(uniprot_info, Protein_id, how='inner')
    need_drug_info.to_csv('datasets/DTI/drug_smiles.csv', index=False)
    need_protein_info.to_csv('datasets/DTI/protein_sequence.csv', index=False)


# --------------------------------------------------get DTI dataset----------------------------------------------------
# get known DTIs from DrugBank dataset
get_drugbank_dti()
# filter DrugBank dataset with needed drugs and needed proteins
filter_drugbank_dti()

# --------------------------------------------------get CPI dataset----------------------------------------------------
# get all cpi activity data from chembl database and bindingdb database
get_cpi()
# get CPI dataset Positive samples and Negative samples according to threshold
get_P_N()
# get pubchem compound info with unique CanonicalSMILES (MolecularWeight<1000) and uniprot protein with unique Sequence
get_info()
# get human PPI data and protein GO data
get_GO_PPI()
# filter CPI dataset with needed drugs and needed proteins
fileter_cpi()
# filtered the CPI dataset by ensuring that each compound and protein is present in both positive and negative samples
fileter_cpi_with_bi_compound_protein()

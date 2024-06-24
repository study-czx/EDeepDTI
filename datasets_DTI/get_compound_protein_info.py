import pandas as pd
import pubchempy as pcp
import datasets_DTI.funcs as funcs
from rdkit import Chem

# 仅保留可以通过RDKit计算出分子描述符的SMILES字符串
def keep_useful_smiles(data_smiles):
    drug_column, smiles_column = data_smiles.columns[0], data_smiles.columns[1]
    keep_id, error_id = [], []
    id_list, smiles_list = data_smiles.iloc[:, 0].tolist(), data_smiles.iloc[:, 1].tolist()
    print('start cal fingerprint')
    for i in range(len(id_list)):
        this_id = id_list[i]
        this_smiles = smiles_list[i]
        m = Chem.MolFromSmiles(this_smiles)
        try:
            RDK = Chem.RDKFingerprint(m)
            keep_id.append(this_id)
        except:
            print('error id (RDKit): ', this_id, this_smiles)
            error_id.append(this_id)
    print(len(error_id))
    keep_id_df = pd.DataFrame(keep_id, columns=[drug_column])
    # error_id_df = pd.DataFrame(error_id, columns=[drug_column])
    # error_id_df.to_csv('processed_data/error_id.csv')
    need_drug_smiles = pd.merge(data_smiles, keep_id_df, how='inner')
    return need_drug_smiles


def get_drug_info():
    Drugbank_link = pd.read_csv('origin_data/DrugBank/structure links.csv')
    Drugbank_smiles = Drugbank_link[['DrugBank ID', 'SMILES']]
    Drugbank_smiles = Drugbank_smiles.dropna()
    Drugbank_smiles = keep_useful_smiles(Drugbank_smiles)
    Drugbank_info = Drugbank_smiles.drop_duplicates(subset='SMILES', keep='first')
    return Drugbank_info


def get_compound_info(all_CPI_data):
    PubChem_id = all_CPI_data[['PubChem_id']].astype(int).sort_values(by='PubChem_id', ascending=True)
    PubChem_id = PubChem_id.drop_duplicates().reset_index(drop=True)
    n = len(PubChem_id)
    id = []
    for i in range(n):
        id.append(PubChem_id['PubChem_id'][i])
    print(len(id))
    weight = pcp.get_properties(identifier=id, properties="MolecularWeight", as_dataframe=True)
    print(weight)
    SMILES = pcp.get_properties(identifier=id, properties="CanonicalSMILES", as_dataframe=True)
    print(SMILES)
    Pubchem_info = pd.merge(weight, SMILES, left_index=True, right_index=True)
    Pubchem_info['MolecularWeight'] = Pubchem_info['MolecularWeight'].astype(float)
    print(len(Pubchem_info))
    Pubchem_with_1000 = Pubchem_info[Pubchem_info['MolecularWeight'] < 1000].reset_index()
    print(len(Pubchem_with_1000))
    Pubchem_new = keep_useful_smiles(Pubchem_with_1000[['CID', 'CanonicalSMILES']])
    print(len(Pubchem_new))
    Pubchem_output = Pubchem_new.drop_duplicates(subset='CanonicalSMILES', keep='first')
    print(len(Pubchem_output))
    return Pubchem_output


def get_uniprot_info(all_CPI_data):
    uniprot_protein = funcs.get_uniprot_review_human()
    uniprot_protein = uniprot_protein[['Uniprot_id', 'Sequence']]
    Uniprot_id = all_CPI_data[['Uniprot_id']].sort_values(by='Uniprot_id', ascending=True)
    Uniprot_id = Uniprot_id.drop_duplicates().reset_index(drop=True)
    need_uniprot_info = pd.merge(Uniprot_id, uniprot_protein, how='inner', on='Uniprot_id')
    uniprot_output = need_uniprot_info.drop_duplicates(subset='Sequence', keep='first')
    return uniprot_output


def get_other_info():
    Uniprot_MF, Uniprot_BP, Uniprot_CC, Uniprot_id = funcs.get_GO_anno()
    print(len(Uniprot_id))
    PPI_data, PPI_id = funcs.get_PPI_data()
    print(len(PPI_id))

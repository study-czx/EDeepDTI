import pandas as pd
import datasets_DTI.funcs as funcs


def get_drugbank_DTI():
    # get DrugBank DTI
    drugbank_data = pd.read_csv('origin_data/DrugBank/Drugbank_DTI.csv')
    drugbank_data = drugbank_data[drugbank_data['Species'] == 'Humans']
    drugbank_data = drugbank_data[['Drug IDs', 'UniProt ID']]
    drugbank_data_new = pd.DataFrame(columns=['Drugbank_ID', 'Uniprot_id'])
    k = 0
    for i in range(len(drugbank_data)):
        drug_ids = drugbank_data.iloc[i, 0]
        uniprot_id = drugbank_data.iloc[i, 1]
        drug_list = drug_ids.split('; ')
        for j in range(len(drug_list)):
            drugbank_data_new.at[k, 'Drugbank_ID'] = drug_list[j]
            drugbank_data_new.at[k, 'Uniprot_id'] = uniprot_id
            k = k + 1
    drugbank_data_new = drugbank_data_new.drop_duplicates()
    drugbank_data_sorted = drugbank_data_new.sort_values(by='Drugbank_ID').reset_index(drop=True)
    return drugbank_data_sorted


def get_chembl_DTI():
    chembl_data = pd.read_csv('origin_data/ChEMBL/Drug Mechanisms.tsv', sep='\t')
    need_data = chembl_data.iloc[:, [0, 10, 14]]

    need_data = need_data[need_data['Target Organism'] == 'Homo sapiens']
    need_data = need_data.drop(columns=['Target Organism'])

    need_data.columns = ['Molecule ChEMBL ID', 'Target ChEMBL ID']
    # print(Chembl_DTI.loc[0])

    print('length of human data', len(need_data))
    need_data = need_data.sort_values(by='Molecule ChEMBL ID')
    return need_data


# DTI_DrugBank = get_drugbank_DTI()
# DTI_DrugBank.to_csv('DTI_data/DrugBank_DTI.csv', index=False)

# DTI_Chembl = get_chembl_DTI()
# DTI_Chembl.to_csv('DTI_data/ChEMBL_DTI.csv', index=False)

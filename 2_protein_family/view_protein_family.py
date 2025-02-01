import pandas as pd



family_list = ['Enzymes', 'GPCR', 'NR', 'IC', 'Others']
datasets = ['DTI', 'CPI_all', 'CPI']
for dataset in datasets:
    print(dataset)
    if dataset == 'DTI':
        all_protein_ids = pd.read_csv('../datasets_DTI/datasets/DTI/protein_id.csv')
        protein_family_info = pd.read_csv('DTI_family_info.csv')
    elif dataset == 'CPI_all':
        all_protein_ids = pd.read_csv('../datasets_DTI/datasets/CPI/all_protein_id.csv')
        protein_family_info = pd.read_csv('CPI_family_info.csv')
    else:
        all_protein_ids = pd.read_csv('../datasets_DTI/datasets/CPI/protein_id.csv')
        protein_family_info = pd.read_csv('CPI_family_info.csv')

    all_protein_ids.columns = ['Uniprot_id']
    protein_family_info = protein_family_info[protein_family_info['Uniprot_id'].isin(all_protein_ids['Uniprot_id'])]

    print('number of Enzymes:', len(protein_family_info[protein_family_info['Family'] == 'Enzymes']))
    print('number of GPCR:', len(protein_family_info[protein_family_info['Family'] == 'GPCR']))
    print('number of NR:', len(protein_family_info[protein_family_info['Family'] == 'NR']))
    print('number of IC:', len(protein_family_info[protein_family_info['Family'] == 'IC']))
    print('number of Others:', len(protein_family_info[protein_family_info['Family'] == 'Others']))

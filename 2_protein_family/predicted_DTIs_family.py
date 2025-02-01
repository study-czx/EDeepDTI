import pandas as pd



family_list = ['Enzymes', 'GPCR', 'NR', 'IC', 'Others']
dataset = 'DTI'

print(dataset)

all_protein_ids = pd.read_csv('../datasets_DTI/datasets/DTI/protein_id.csv')
protein_family_info = pd.read_csv('DTI_family_info.csv')

all_protein_ids.columns = ['Uniprot_id']
protein_family_info = protein_family_info[protein_family_info['Uniprot_id'].isin(all_protein_ids['Uniprot_id'])]



print('number of Enzymes:', len(protein_family_info[protein_family_info['Family'] == 'Enzymes']))
print('number of GPCR:', len(protein_family_info[protein_family_info['Family'] == 'GPCR']))
print('number of NR:', len(protein_family_info[protein_family_info['Family'] == 'NR']))
print('number of IC:', len(protein_family_info[protein_family_info['Family'] == 'IC']))
print('number of Others:', len(protein_family_info[protein_family_info['Family'] == 'Others']))

match_DTIs = pd.read_csv('../case study/TP_DTI.csv')
match_DTIs_info = pd.merge(match_DTIs, protein_family_info, on=['Uniprot_id'], how='inner')
print('number of Enzymes:', len(match_DTIs_info[match_DTIs_info['Family'] == 'Enzymes']))
print('number of GPCR:', len(match_DTIs_info[match_DTIs_info['Family'] == 'GPCR']))
print('number of NR:', len(match_DTIs_info[match_DTIs_info['Family'] == 'NR']))
print('number of IC:', len(match_DTIs_info[match_DTIs_info['Family'] == 'IC']))
print('number of Others:', len(match_DTIs_info[match_DTIs_info['Family'] == 'Others']))
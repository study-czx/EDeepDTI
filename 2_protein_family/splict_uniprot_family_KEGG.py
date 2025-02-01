import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

datasets = ['DTI', 'CPI']
# family_map = {'hsa01000':'Enzymes', 'hsa04030':'GPCR', 'hsa03310':'NR', 'hsa04040':'IC'}

family_names = ['Enzymes', 'GPCR', 'NR', 'IC']
family_codes = ['hsa01000', 'hsa04030', 'hsa03310', 'hsa04040']

for dataset in datasets:
    print('dataset:', dataset)
    uniprot_family = pd.read_csv('KEGG/' + dataset + '/protein_family_info.csv')
    uniprot_family = uniprot_family.drop_duplicates().reset_index(drop=True)
    uniprot_family['Category'] = 'Unclassified'
    protein_family_need = uniprot_family[uniprot_family['family'].isin(family_codes)]

    enzymes_protein = protein_family_need[protein_family_need['family'] == 'hsa01000']
    enzymes_protein['Category'] = 'Enzymes'
    enzymes_protein = enzymes_protein[['Uniprot_id', 'Category']].drop_duplicates().reset_index(drop=True)
    gpcr_protein = protein_family_need[protein_family_need['family'] == 'hsa04030']
    gpcr_protein['Category'] = 'GPCR'
    gpcr_protein = gpcr_protein[['Uniprot_id', 'Category']].drop_duplicates().reset_index(drop=True)
    IC_protein = protein_family_need[protein_family_need['family'] == 'hsa04040']
    IC_protein['Category'] = 'IC'
    IC_protein = IC_protein[['Uniprot_id', 'Category']].drop_duplicates().reset_index(drop=True)
    NR_protein = protein_family_need[protein_family_need['family'] == 'hsa03310']
    NR_protein['Category'] = 'NR'
    NR_protein = NR_protein[['Uniprot_id', 'Category']].drop_duplicates().reset_index(drop=True)

    print('number of Enzymes:', len(enzymes_protein))
    print('number of GPCR:', len(gpcr_protein))
    print('number of NR:', len(NR_protein))
    print('number of IC:', len(IC_protein))

    four_family_protein_info = pd.concat([enzymes_protein, gpcr_protein, NR_protein, IC_protein]).reset_index(drop=True)
    save_path = 'KEGG/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    four_family_protein_info.to_csv(save_path + 'protein_family.csv', index=False)




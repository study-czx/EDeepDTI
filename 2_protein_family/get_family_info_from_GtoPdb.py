import os

import pandas as pd

all_protein_info = pd.read_csv('GtoPdb/GtoPdb.csv', comment='#', skiprows=1)

all_protein_family = all_protein_info[['Type', 'Human SwissProt']]
all_protein_family.columns = ['Category', 'Uniprot_id']

datasets = ['DTI', 'CPI']
for dataset in datasets:
    print('dataset:', dataset)
    if dataset == 'DTI':
        protein_ids = pd.read_csv('../datasets_DTI/datasets/'+dataset+'/protein_id.csv')
    else:
        protein_ids = pd.read_csv('../datasets_DTI/datasets/'+dataset+'/all_protein_id.csv')

    protein_family = pd.merge(protein_ids, all_protein_family, on='Uniprot_id', how='inner')

    enzymes_protein = protein_family[protein_family['Category'] == 'enzyme'].drop_duplicates().reset_index(drop=True)
    gpcr_protein = protein_family[protein_family['Category'] == 'gpcr'].drop_duplicates().reset_index(drop=True)
    IC_protein = protein_family[protein_family['Category'].isin(['lgic', 'vgic', 'other_ic'])].drop_duplicates().reset_index(drop=True)
    NR_protein = protein_family[protein_family['Category'] == 'nhr'].drop_duplicates().reset_index(drop=True)

    enzymes_protein['Category'] = 'Enzymes'
    gpcr_protein['Category'] = 'GPCR'
    NR_protein['Category'] = 'NR'
    IC_protein['Category'] = 'IC'
    four_family_protein_info = pd.concat([enzymes_protein, gpcr_protein, NR_protein, IC_protein]).reset_index(drop=True)
    # print(four_family_protein_info)
    print('number of Enzymes:', len(enzymes_protein))
    print('number of GPCR:', len(gpcr_protein))
    print('number of NR:', len(NR_protein))
    print('number of IC:', len(IC_protein))

    save_path = 'GtoPdb/'+dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    four_family_protein_info.to_csv(save_path+'protein_family.csv', index=False)



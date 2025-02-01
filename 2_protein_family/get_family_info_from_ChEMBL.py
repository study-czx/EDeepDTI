import pandas as pd
import os

def splict_dataset(data):
    all_proteins = []
    for i in range(len(data)):
        this_protein = str(data['UniProt Accessions'][i])
        if len(this_protein) > 0:
            protein_list = this_protein.split('|')
            for pro in protein_list:
                all_proteins.append(pro)
    output_proteins = sorted(list(set(all_proteins)))
    df_output_proteins = pd.DataFrame(output_proteins, columns=['Uniprot_id'])
    return df_output_proteins


enzymy_protein_info = pd.read_csv('ChEMBL/ChEMBL_Enzyme.tsv', sep='\t', usecols=['UniProt Accessions'])
GPCR_protein_info = pd.read_csv('ChEMBL/ChEMBL_GPCR.tsv', sep='\t', usecols=['UniProt Accessions'])
IC_protein_info = pd.read_csv('ChEMBL/ChEMBL_IC.tsv', sep='\t', usecols=['UniProt Accessions'])
NR_protein_info = pd.read_csv('ChEMBL/ChEMBL_NR.tsv', sep='\t', usecols=['UniProt Accessions'])


enzymy_protein_split = splict_dataset(enzymy_protein_info)
GPCR_protein_split = splict_dataset(GPCR_protein_info)
IC_protein_split = splict_dataset(IC_protein_info)
NR_protein_split = splict_dataset(NR_protein_info)


datasets = ['DTI', 'CPI']
for dataset in datasets:
    print('dataset:', dataset)
    if dataset == 'DTI':
        protein_ids = pd.read_csv('../datasets_DTI/datasets/'+dataset+'/protein_id.csv')
    elif dataset == 'CPI':
        protein_ids = pd.read_csv('../datasets_DTI/datasets/'+dataset+'/all_protein_id.csv')

    enzymes_protein = pd.merge(protein_ids, enzymy_protein_split, on='Uniprot_id', how='inner')
    gpcr_protein = pd.merge(protein_ids, GPCR_protein_split, on='Uniprot_id', how='inner')
    IC_protein = pd.merge(protein_ids, IC_protein_split, on='Uniprot_id', how='inner')
    NR_protein = pd.merge(protein_ids, NR_protein_split, on='Uniprot_id', how='inner')

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

    save_path = 'ChEMBL/'+dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    four_family_protein_info.to_csv(save_path+'protein_family.csv', index=False)

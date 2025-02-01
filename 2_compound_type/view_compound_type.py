import pandas as pd

all_compounds = pd.read_csv('../datasets_DTI/datasets/CPI/all_compound_id.csv')

print(all_compounds)

PubChem_ChemBL_map = pd.read_table('src1src22.txt', sep='\t')
PubChem_ChemBL_map.columns = ['ChEMBl_id', 'PubChem_id']

all_compounds_new = pd.merge(all_compounds, PubChem_ChemBL_map, on='PubChem_id', how='left')
print(all_compounds_new[all_compounds_new['ChEMBl_id'].notna()])

ChEMBl_compounds1 = pd.read_csv('ChEMBL_compound1.tsv', sep='\t', usecols=['ChEMBL ID', 'Max Phase']).reset_index(
    drop=True)
ChEMBl_compounds2 = pd.read_csv('ChEMBL_compound2.tsv', sep='\t', usecols=[0, 4], header=None).reset_index(drop=True)
ChEMBl_compounds2.columns = ChEMBl_compounds1.columns
ChEMBl_compounds = pd.concat([ChEMBl_compounds1, ChEMBl_compounds2], ignore_index=True).reset_index(drop=True)

ChEMBl_compounds.columns = ['ChEMBl_id', 'Max Phase']

all_compounds_new = pd.merge(all_compounds_new, ChEMBl_compounds, on='ChEMBl_id', how='left')
print(all_compounds_new)

print(all_compounds_new[all_compounds_new['Max Phase'] == 4])
print(all_compounds_new[all_compounds_new['Max Phase'].isin([-1, 0.5, 1, 2, 3])])
print(all_compounds_new[all_compounds_new['Max Phase'].isna()])



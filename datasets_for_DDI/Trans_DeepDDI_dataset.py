import pandas as pd


types = ['train', 'valid', 'test']

drug_list_deep = pd.read_csv('datasets_for_DDI/drug_list_deep.csv')
drug_list_deep1 = drug_list_deep.copy()
drug_list_deep2 = drug_list_deep.copy()
# print(drug_list_deep1)

drug_list_deep1.columns = ['drugbank_id_1', 'smiles_1']
drug_list_deep2.columns = ['drugbank_id_2', 'smiles_2']

for type in types:
    DeepDDI_data = pd.read_csv('datasets_for_DDI/DeepDDI_'+type+'_o.csv')
    new_DeepDDI_data = pd.merge(DeepDDI_data, drug_list_deep1, on='smiles_1')
    # print(new_DeepDDI_data)
    new_DeepDDI_data2 = pd.merge(new_DeepDDI_data, drug_list_deep2, on='smiles_2')
    # print(new_DeepDDI_data2)
    output_data = new_DeepDDI_data2[['drugbank_id_1','drugbank_id_2','label']]
    output_data.to_csv('datasets_for_DDI/DeepDDI_'+type+'.csv', index=False)
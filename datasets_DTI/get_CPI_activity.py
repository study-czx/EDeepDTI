import pandas as pd
import os
import warnings
import datasets_DTI.funcs as funcs

warnings.filterwarnings('ignore')

def get_chembl_activity(target_map, molecule_map, uniprot_protein):
    folder_path = 'origin_data/ChEMBL/ChEMBL_activity'
    files = sorted(os.listdir(folder_path))
    files.sort(key=lambda x: int(x.split('.csv')[0]))
    all_data = pd.DataFrame()
    for file in files:
        if file == '1.csv':
            this_data = pd.read_csv(folder_path + '/' + file, sep=";", header=0, low_memory=False)
        else:
            this_data = pd.read_csv(folder_path + '/' + file, sep=";", header=None, low_memory=False)
        # print(this_data.loc[0])
        need_data = this_data.iloc[:, [0, 34, 8, 9, 10, 11]]
        need_data.columns = ['Molecule ChEMBL ID', 'Target ChEMBL ID', 'Standard Type',
                             'Standard Relation', 'Standard Value', 'Standard Units']
        # remove ''
        need_data['Standard Relation'] = need_data['Standard Relation'].str.strip("'")

        # print('length of raw data: ', len(need_data))
        need_data = need_data[need_data['Standard Units'] == 'nM']
        # print('length of nM data: ', len(need_data))
        need_data = need_data[need_data['Standard Value'] > 0]
        # print('length of value > 0 data: ', len(need_data))

        filtered_data = need_data[need_data['Standard Type'].isin(['IC50', 'EC50', 'Ki', 'Kd'])]
        # print('length of IC50_EC50_Ki_Kd data: ', len(filtered_data))
        filtered_data2 = filtered_data.dropna(subset=['Standard Value'])
        # print('length of has_value data: ', len(filtered_data2))
        filtered_data3 = pd.merge(filtered_data2, molecule_map, on='Molecule ChEMBL ID')
        # print('length of pubchem data: ', len(filtered_data3))
        filtered_data4 = pd.merge(filtered_data3, target_map, on='Target ChEMBL ID')
        # print('length of uniprot data: ', len(filtered_data4))
        filtered_data5 = pd.merge(filtered_data4, uniprot_protein, how='inner', on='Uniprot_id')
        # print('length of reviewed uniprot data: ', len(filtered_data5))
        filtered_data6 = filtered_data5[filtered_data5['Standard Relation'].isin(['>=', '<=', '>', '<', '='])]
        # print('length of > >= = < <= data: ', len(filtered_data6))


        if all_data.empty:
            all_data = filtered_data6
        else:
            all_data = pd.concat([all_data, filtered_data6], axis=0)

    # print('all data number: ', len(all_data))
    all_data.drop('Standard Units', axis=1, inplace=True)
    all_data = all_data.drop_duplicates()
    # print('unique data number: ', len(all_data))
    all_data = all_data.sort_values(by='Molecule ChEMBL ID')
    return all_data
    # print(set(list(all_data['Standard Relation'])))


def process_value(value):
    if value.startswith('>') or value.startswith('<'):
        return value[0], value[1:]
    else:
        return '=', value


def get_bindingdb_data(uniprot_protein):
    # 2812698
    bindingdb_data = pd.read_csv('origin_data/BindingDB/BindingDB_All_202401.tsv', sep='\t',
                                 usecols=['PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain',
                                          'IC50 (nM)', 'EC50 (nM)', 'Ki (nM)', 'Kd (nM)'], low_memory=False)
    # print('raw data: ', len(bindingdb_data))
    bindingdb_data = bindingdb_data.dropna(subset=['PubChem CID'])
    # print('with pubchem data: ', len(bindingdb_data))
    bindingdb_data = bindingdb_data.dropna(subset=['UniProt (SwissProt) Primary ID of Target Chain'])
    # print('with uniprot data: ', len(bindingdb_data))

    IC50_data = bindingdb_data.dropna(subset=['IC50 (nM)'])
    IC50_data = IC50_data[['PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain', 'IC50 (nM)']]

    EC50_data = bindingdb_data.dropna(subset=['EC50 (nM)'])
    EC50_data = EC50_data[['PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain', 'EC50 (nM)']]

    Ki_data = bindingdb_data.dropna(subset=['Ki (nM)'])
    Ki_data = Ki_data[['PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain', 'Ki (nM)']]

    Kd_data = bindingdb_data.dropna(subset=['Kd (nM)'])
    Kd_data = Kd_data[['PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain', 'Kd (nM)']]

    IC50_data.columns = ['PubChem_id', 'Uniprot_id', 'Standard Value']
    IC50_data['Standard Type'] = 'IC50'

    EC50_data.columns = ['PubChem_id', 'Uniprot_id', 'Standard Value']
    EC50_data['Standard Type'] = 'EC50'

    Ki_data.columns = ['PubChem_id', 'Uniprot_id', 'Standard Value']
    Ki_data['Standard Type'] = 'Ki'

    Kd_data.columns = ['PubChem_id', 'Uniprot_id', 'Standard Value']
    Kd_data['Standard Type'] = 'Kd'

    All_data = pd.concat([IC50_data, EC50_data, Ki_data, Kd_data])
    # print('all data number: ', len(All_data))

    filtered_data = All_data[All_data['Uniprot_id'].isin(uniprot_protein['Uniprot_id'])].reset_index(drop=True)
    # print('human protein number: ', len(filtered_data))

    filtered_data['Standard Relation'] = ''
    need_data = filtered_data.copy()

    need_data['Standard Relation'], need_data['Standard Value'] = zip(
        *filtered_data['Standard Value'].apply(process_value))

    need_data['Standard Value'] = need_data['Standard Value'].astype(float)
    need_data = need_data[need_data['Standard Value'] > 0]
    # print('value > 0 number: ', len(filtered_data))

    need_data['PubChem_id'] = need_data['PubChem_id'].astype(int)

    need_data = need_data.drop_duplicates()
    # print('unique data number: ', len(need_data))
    need_data = need_data.sort_values(by='PubChem_id')
    return need_data

def get_cpi_data():
    print('start get chembl activity data')
    chembl_target_map = funcs.get_chembl_target_data()
    chembl_molecule_map = funcs.get_chembl_pubchem_map()
    uniprot_protein = funcs.get_uniprot_review_human()
    uniprot_protein = uniprot_protein[['Uniprot_id']]
    chembl_cpi = get_chembl_activity(chembl_target_map, chembl_molecule_map, uniprot_protein)
    print('finish get chembl activity data')

    print('start get bindingdb activity data')
    bindingdb_data = get_bindingdb_data(uniprot_protein)
    print('finish get bindingdb activity data')
    return chembl_cpi, bindingdb_data


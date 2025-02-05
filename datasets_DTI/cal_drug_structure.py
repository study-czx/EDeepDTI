import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import datasets_DTI.funcs as funcs


def get_RDKit_finger(num_drugs, mols):
    RDK_finger = []
    for x in mols:
        RDK = Chem.RDKFingerprint(x)
        RDK_out = RDK.ToBitString()
        RDK_finger.append(RDK_out)
    RDK_matrix = np.zeros(shape=(num_drugs, 2048))
    for i in range(num_drugs):
        for j in range(2048):
            RDK_matrix[i][j] = int(RDK_finger[i][j])
    RDK_frame = pd.DataFrame(RDK_matrix, dtype=int)
    return RDK_frame


def get_ECFP_FCFP_finger(mols):
    n_Morgen = 1024
    # n_Morgen = 2048
    ECFP4_finger, FCFP4_finger = [], []
    for x in mols:
        ECFP4 = AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=n_Morgen)
        FCFP4 = AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=n_Morgen, useFeatures=True)
        ECFP4_finger.append(list(str(ECFP4.ToBitString())))
        FCFP4_finger.append(list(str(FCFP4.ToBitString())))
    ECFP4_frame = pd.DataFrame(ECFP4_finger)
    FCFP4_frame = pd.DataFrame(FCFP4_finger)
    return ECFP4_frame, FCFP4_frame


def get_MACCS_finger(mols):
    MACCS_finger = []
    for x in mols:
        MACCS = MACCSkeys.GenMACCSKeys(x)
        MACCS_finger.append(list(str(MACCS.ToBitString())))
    MACCS_frame = pd.DataFrame(MACCS_finger)
    MACCS_frame = MACCS_frame.drop(MACCS_frame.columns[0], axis=1)
    return MACCS_frame


def get_fingerprint(structures, type):
    name1, name2 = 'Drugbank_id', 'SMILES'
    if type == 'DTI':
        name1, name2 = name1, name2
    elif type == 'CPI':
        name1, name2 = 'PubChem_id', 'CanonicalSMILES'
    else:
        name1, name2 = 'drug', 'smiles'
    # get id list and smiles list
    id = []
    smiles = []
    for i in range(len(structures)):
        id.append(structures[name1][i])
        smiles.append(structures[name2][i])


    num_drugs = len(id)

    # get mols by RDKit
    mols = []
    for k in smiles:
        m = Chem.MolFromSmiles(k)
        mols.append(m)

    ECFP4_frame, FCFP4_frame = get_ECFP_FCFP_finger(mols)
    print(ECFP4_frame)
    print(FCFP4_frame)
    MACCS_frame = get_MACCS_finger(mols)
    print(MACCS_frame)
    return MACCS_frame, ECFP4_frame, FCFP4_frame


def write_csv(MACCS_data, RDK_data, ECFP4_data, FCFP4_data, type):
    path = 'datasets/' + type + '/drug_finger/'
    funcs.Make_path(path)
    MACCS_data.to_csv(path + 'MACCS.csv', index=False)
    RDK_data.to_csv(path + 'RDKit.csv', index=False)
    ECFP4_data.to_csv(path + 'ECFP4.csv', index=False)
    FCFP4_data.to_csv(path + 'FCFP4.csv', index=False)


def run_cal_drug_discriptor(dataset):
    if dataset == 'DTI':
        Drug_structure = pd.read_csv('datasets/DTI/drug_smiles.csv', sep=',', dtype=str)
        DTI_MACCS_frame,  DTI_ECFP4_frame, DTI_FCFP4_frame = get_fingerprint(Drug_structure, type='DTI')
        write_csv(DTI_MACCS_frame, DTI_ECFP4_frame, DTI_FCFP4_frame, type='DTI')
    elif dataset == 'CPI':
        Compound_structure = pd.read_csv('datasets/CPI/all_compound_smiles.csv', sep=',', dtype=str)
        CPI_MACCS_frame, CPI_ECFP4_frame, CPI_FCFP4_frame = get_fingerprint(Compound_structure, type='CPI')
        write_csv(CPI_MACCS_frame, CPI_ECFP4_frame, CPI_FCFP4_frame, type='CPI')
    elif dataset == 'Davis' or dataset == 'KIBA':
        data_type = dataset + '_5fold'
        Compound_structure = pd.read_csv('datasets/'+data_type+'/Drug.csv', sep=',', dtype=str)
        CPI_MACCS_frame,  CPI_ECFP4_frame, CPI_FCFP4_frame = get_fingerprint(Compound_structure, type='DK')
        write_csv(CPI_MACCS_frame, CPI_ECFP4_frame, CPI_FCFP4_frame, type=data_type)


datasets = ['DTI', 'CPI', 'Davis', 'KIBA']
for dataset in datasets:
    print(dataset)
    run_cal_drug_discriptor(dataset)



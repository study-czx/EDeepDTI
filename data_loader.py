import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import funcs
import torch

def Get_feature_numbers(data_type, input_type):
    n_dr_feats, n_p_feats = 0, 0
    if input_type == 'd':
        n_dr_feats, n_p_feats = 4, 5
    elif input_type == 'e':
        n_dr_feats, n_p_feats = 9, 6
    elif input_type == 's':
        if data_type == 'DTI':
            n_dr_feats, n_p_feats = 5, 6
        elif data_type == 'CPI':
            n_dr_feats, n_p_feats = 9, 6
        else:
            n_dr_feats, n_p_feats = 4, 6
    # print('number of drug feature types: ', n_dr_feats)
    # print('number of protein feature types: ', n_p_feats)
    return n_dr_feats, n_p_feats

def Get_data_path(data_type):
    dir_path = 'datasets_DTI/datasets/' + data_type
    feature_path_dr = dir_path + '/drug_finger/'
    feature_path_p = dir_path + '/protein_descriptor/'
    sim_feature_path_dr = dir_path + '/drug_sim/'
    sim_feature_path_p = dir_path + '/protein_sim/'
    emb_path_dr = dir_path + '/drug_embedding/'
    emb_path_p = dir_path + '/protein_embedding/'
    return feature_path_dr, feature_path_p, sim_feature_path_dr, sim_feature_path_p, emb_path_dr, emb_path_p


def Get_finger(data_type):
    feature_path_dr, _, _, _, _, _ = Get_data_path(data_type)
    MACCS = np.loadtxt(feature_path_dr + 'MACCS.csv', dtype=float, delimiter=',', skiprows=1)
    Pubchem = np.loadtxt(feature_path_dr + 'PubChem.csv', dtype=float, delimiter=',', skiprows=1)
    # RDK = np.loadtxt(feature_path_dr + 'RDKit.csv', dtype=float, delimiter=',', skiprows=1)
    ECFP4 = np.loadtxt(feature_path_dr + 'ECFP4.csv', dtype=float, delimiter=',', skiprows=1)
    FCFP4 = np.loadtxt(feature_path_dr + 'FCFP4.csv', dtype=float, delimiter=',', skiprows=1)
    Dr_finger = {'maccs': MACCS, 'pubchem': Pubchem, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
    return Dr_finger


def Get_seq(data_type):
    _, feature_path_p, _, _, _, _ = Get_data_path(data_type)
    scaler = MinMaxScaler()

    TPC = np.loadtxt(feature_path_p + 'TPC.csv', dtype=float, delimiter=',')
    PAAC = np.loadtxt(feature_path_p + 'PAAC.csv', dtype=float, delimiter=',')
    PAAC = scaler.fit_transform(PAAC)

    KSCTriad = np.loadtxt(feature_path_p + 'KSCTriad.csv', dtype=float, delimiter=',')
    CKSAAP = np.loadtxt(feature_path_p + 'CKSAAP.csv', dtype=float, delimiter=',')
    CTDC = np.loadtxt(feature_path_p + 'CTDC.csv', dtype=float, delimiter=',')
    CTDT = np.loadtxt(feature_path_p + 'CTDT.csv', dtype=float, delimiter=',')
    CTDD = np.loadtxt(feature_path_p + 'CTDD.csv', dtype=float, delimiter=',')
    CTDD = scaler.fit_transform(CTDD)

    CTD = np.concatenate((CTDC, CTDT, CTDD), axis=1)
    P_seq = {'PAAC': PAAC, 'KSCTriad': KSCTriad, 'TPC': TPC, 'CKSAAP': CKSAAP, 'CTD': CTD}
    return P_seq


def Get_drug_sim(data_type):
    _, _, sim_feature_path_dr, _, _, _ = Get_data_path(data_type)
    MACCS = np.loadtxt(sim_feature_path_dr + 'MACCS.csv', dtype=float, delimiter=',')
    Pubchem = np.loadtxt(sim_feature_path_dr + 'PubChem.csv', dtype=float, delimiter=',')
    # RDK = np.loadtxt(sim_feature_path_dr + 'RDKit.csv', dtype=float, delimiter=',')
    ECFP4 = np.loadtxt(sim_feature_path_dr + 'ECFP4.csv', dtype=float, delimiter=',')
    FCFP4 = np.loadtxt(sim_feature_path_dr + 'FCFP4.csv', dtype=float, delimiter=',')
    if data_type == 'DTI':
        DDI_sim = np.loadtxt(sim_feature_path_dr + 'DDI.csv', dtype=float, delimiter=',')
        Dr_sim = {'maccs': MACCS, 'pubchem': Pubchem, 'ecfp4': ECFP4, 'fcfp4': FCFP4, 'DDI': DDI_sim}
        return Dr_sim
    else:
        Dr_sim = {'maccs': MACCS, 'pubchem': Pubchem, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
        return Dr_sim


def Get_protein_sim(data_type):
    _, _, _, sim_feature_path_p, _, _ = Get_data_path(data_type)
    seq_sim = np.loadtxt(sim_feature_path_p + 'seq.csv', dtype=float, delimiter=',', skiprows=1)
    PPI_a_sim = np.loadtxt(sim_feature_path_p + 'PPI_a.csv', dtype=float, delimiter=',', skiprows=0)
    PPI_t_sim = np.loadtxt(sim_feature_path_p + 'PPI_t.csv', dtype=float, delimiter=',', skiprows=0)
    MF_sim = np.loadtxt(sim_feature_path_p + 'MF.csv', dtype=float, delimiter=',', skiprows=1)
    BP_sim = np.loadtxt(sim_feature_path_p + 'BP.csv', dtype=float, delimiter=',', skiprows=1)
    CC_sim = np.loadtxt(sim_feature_path_p + 'CC.csv', dtype=float, delimiter=',', skiprows=1)
    P_sim = {'seq': seq_sim, 'PPI_a': PPI_a_sim, 'PPI2': PPI_t_sim, 'MF': MF_sim, 'BP': BP_sim, 'CC': CC_sim}
    return P_sim


def Get_drug_embedding(data_type):
    _, _, _, _, emb_feature_path_dr, _ = Get_data_path(data_type)
    chemberta2 = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_emb.csv', dtype=float, delimiter=',')
    chemberta2_mtr = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_emb_MTR.csv', dtype=float, delimiter=',')
    molformer = np.loadtxt(emb_feature_path_dr + 'Molformer_emb.csv', dtype=float, delimiter=',')
    grover = np.loadtxt(emb_feature_path_dr + 'grover.csv', dtype=float, delimiter=',')
    kpgt = np.loadtxt(emb_feature_path_dr + 'kpgt_emb.csv', dtype=float, delimiter=',')

    # chemberta2_max = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_emb_max.csv', dtype=float, delimiter=',')
    # chemberta2_mtr_max = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_emb_MTR_max.csv', dtype=float, delimiter=',')
    # grover_max = np.loadtxt(emb_feature_path_dr + 'grover_max.csv', dtype=float, delimiter=',')
    # molformer_max = np.loadtxt(emb_feature_path_dr + 'Molformer_emb_max.csv', dtype=float, delimiter=',')
    # kpgt_max = np.loadtxt(emb_feature_path_dr + 'kpgt_emb_max.csv', dtype=float, delimiter=',')

    Dr_embedding = {'chemberta2': chemberta2, 'chemberta2_mtr': chemberta2_mtr,'molformer': molformer, 'grover': grover, 'kpgt': kpgt}
    return Dr_embedding


def Get_protein_embedding(data_type):
    _, _, _, _, _, emb_feature_path_p = Get_data_path(data_type)
    esm2 = np.loadtxt(emb_feature_path_p + 'ESM2_emb.csv', dtype=float, delimiter=',')
    protein_bert = np.loadtxt(emb_feature_path_p + 'Protein_bert_emb.csv', dtype=float, delimiter=',')
    prottrans = np.loadtxt(emb_feature_path_p + 'ProtTrans_emb.csv', dtype=float, delimiter=',')
    esm2_max = np.loadtxt(emb_feature_path_p + 'ESM2_emb_max.csv', dtype=float, delimiter=',')
    protein_bert_max = np.loadtxt(emb_feature_path_p + 'Protein_bert_emb_max.csv', dtype=float, delimiter=',')
    prottrans_max = np.loadtxt(emb_feature_path_p + 'ProtTrans_emb_max.csv', dtype=float, delimiter=',')

    P_embedding = {'prottrans': prottrans, 'protein_bert': protein_bert, 'esm2': esm2,
                   'prottrans_max': prottrans_max,'protein_bert_max': protein_bert_max, 'esm2_max': esm2_max}
    return P_embedding


def Get_id(data_type):
    if data_type == 'DTI':
        Drug_id = pd.read_csv('datasets_DTI/datasets/DTI/Drug_id.csv', sep=',', dtype=str)
        Protein_id = pd.read_csv('datasets_DTI/datasets/DTI/protein_id.csv', sep=',', dtype=str)
        Drug_id, Protein_id = Drug_id.iloc[:, 0].tolist(), Protein_id.iloc[:, 0].tolist()
        return Drug_id, Protein_id
    elif data_type == 'CPI':
        Drug_id = pd.read_csv('datasets_DTI/datasets/CPI/all_compound_id.csv', sep=',', dtype=str)
        Protein_id = pd.read_csv('datasets_DTI/datasets/CPI/all_protein_id.csv', sep=',', dtype=str)
        Drug_id, Protein_id = Drug_id.iloc[:, 0].tolist(), Protein_id.iloc[:, 0].tolist()
        return Drug_id, Protein_id
    else:
        Drug_id = pd.read_csv('datasets_DTI/datasets/' + data_type + '/Drug.csv', sep=',', dtype=str)
        Protein_id = pd.read_csv('datasets_DTI/datasets/' + data_type + '/Protein.csv', sep=',', dtype=str)
        Drug_id, Protein_id = Drug_id.iloc[:, 0].tolist(), Protein_id.iloc[:, 0].tolist()
        return Drug_id, Protein_id



def Get_feature(data_type, input_type):
    Drug_id, Protein_id = Get_id(data_type)
    n_drugs, n_proteins = len(Drug_id), len(Protein_id)
    dr_id_map, p_id_map = funcs.id_map(Drug_id), funcs.id_map(Protein_id)

    if input_type == 'd':
        Dr_finger, P_seq = Get_finger(data_type), Get_seq(data_type)
        return dr_id_map, p_id_map, Dr_finger, P_seq
    elif input_type == 'e':
        Dr_finger = Get_finger(data_type)
        Dr_embedding, P_embedding = Get_drug_embedding(data_type), Get_protein_embedding(data_type)
        Dr_finger.update(Dr_embedding)
        return dr_id_map, p_id_map, Dr_finger, P_embedding
    elif input_type == 's':
        if data_type == 'CPI':
            Dr_finger = Get_finger(data_type)
            Dr_embedding, P_embedding = Get_drug_embedding(data_type), Get_protein_embedding(data_type)
            Dr_finger.update(Dr_embedding)
            P_sim = Get_protein_sim(data_type)
            return dr_id_map, p_id_map, Dr_finger, P_sim
        else:
            Dr_sim, P_sim = Get_drug_sim(data_type), Get_protein_sim(data_type)
            return dr_id_map, p_id_map, Dr_sim, P_sim


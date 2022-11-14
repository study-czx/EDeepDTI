import numpy as np
import pandas as pd
dir_path = 'D:/Users/czx/PycharmProjects/2-server-Experiment/'
feature_path_CPI_dr = dir_path + 'CPI_dataset/CPI_feature/drug/'
feature_path_CPI_p = dir_path + 'CPI_dataset/CPI_feature/protein/'
sim_feature_path_CPI_dr = dir_path + 'CPI_dataset/CPI_feature/drug_sim/'
sim_feature_path_CPI_p = dir_path + 'CPI_dataset/CPI_feature/protein_sim/'

def Get_CPI_id():
    Drug_structure = pd.read_csv("./CPI_dataset/SMILES_8165.csv", sep=',', dtype=str)
    Protein_seq = pd.read_csv(r"./CPI_dataset/seq_683.csv", sep=',', dtype=str)
    Drug_id, Protein_id = [], []
    for i in range(len(Drug_structure)):
        Drug_id.append(Drug_structure['CID'][i])
    for j in range(len(Protein_seq)):
        Protein_id.append(Protein_seq['uniprot'][j])
    return Drug_id, Protein_id

def Get_CPI_finger():
    MACCS = np.loadtxt(feature_path_CPI_dr + "MACCS.csv", dtype=float, delimiter=",")
    Pubchem = np.loadtxt(feature_path_CPI_dr + "Pubchem.csv", dtype=float, delimiter=",")
    RDK = np.loadtxt(feature_path_CPI_dr + "RDK_2048.csv", dtype=float, delimiter=",")
    ECFP4 = np.loadtxt(feature_path_CPI_dr + "ECFP4.csv", dtype=float, delimiter=",")
    FCFP4 = np.loadtxt(feature_path_CPI_dr + "FCFP4.csv", dtype=float, delimiter=",")
    Dr_finger = {'maccs':MACCS,'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
    return Dr_finger

def Get_CPI_seq():
    TPC = np.loadtxt(feature_path_CPI_p + "TPC.csv", dtype=float, delimiter=",", skiprows=0)
    PAAC = np.loadtxt(feature_path_CPI_p + "PAAC.csv", dtype=float, delimiter=",", skiprows=0)
    KSCTriad = np.loadtxt(feature_path_CPI_p + "KSCTriad.csv", dtype=float, delimiter=",", skiprows=0)
    CKSAAP = np.loadtxt(feature_path_CPI_p + "CKSAAP.csv", dtype=float, delimiter=",", skiprows=0)
    CTD = np.loadtxt(feature_path_CPI_p + "CTD.csv", dtype=float, delimiter=",", skiprows=0)
    P_seq = {'PAAC': PAAC, 'KSCTriad': KSCTriad, 'TPC': TPC, 'CKSAAP':CKSAAP, 'CTD':CTD}
    return P_seq

def Get_CPI_drug_sim():
    MACCS = np.loadtxt(sim_feature_path_CPI_dr + "MACCS.csv", dtype=float, delimiter=",", skiprows=1)
    Pubchem = np.loadtxt(sim_feature_path_CPI_dr + "Pubchem.csv", dtype=float, delimiter=",", skiprows=1)
    RDK = np.loadtxt(sim_feature_path_CPI_dr + "RDK.csv", dtype=float, delimiter=",", skiprows=1)
    ECFP4 = np.loadtxt(sim_feature_path_CPI_dr + "ECFP4.csv", dtype=float, delimiter=",", skiprows=1)
    FCFP4 = np.loadtxt(sim_feature_path_CPI_dr + "FCFP4.csv", dtype=float, delimiter=",", skiprows=1)
    Dr_sim = {'maccs': MACCS, 'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
    return Dr_sim

def Get_CPI_protein_sim():
    seq_sim = np.loadtxt(sim_feature_path_CPI_p+"seq.csv", dtype=float, delimiter=",", skiprows=1)
    PPI_sim = np.loadtxt(sim_feature_path_CPI_p+"PPI_sim.csv", dtype=float, delimiter=",", skiprows=1)
    PPI2_sim = np.loadtxt(sim_feature_path_CPI_p+"PPI2_sim.csv", dtype=float, delimiter=",", skiprows=1)
    P_D_sim = np.loadtxt(sim_feature_path_CPI_p+"P_D_sim.csv", dtype=float, delimiter=",", skiprows=1)
    MF_sim = np.loadtxt(sim_feature_path_CPI_p+"GO_sim/MF_sim.csv", dtype=float, delimiter=",", skiprows=1)
    BP_sim = np.loadtxt(sim_feature_path_CPI_p + "GO_sim/BP_sim.csv", dtype=float, delimiter=",", skiprows=1)
    CC_sim = np.loadtxt(sim_feature_path_CPI_p + "GO_sim/CC_sim.csv", dtype=float, delimiter=",", skiprows=1)
    P_sim = {'seq': seq_sim, 'PPI': PPI_sim, 'PPI2': PPI2_sim, 'P_D': P_D_sim, 'MF': MF_sim, 'BP': BP_sim, 'CC': CC_sim}
    return P_sim


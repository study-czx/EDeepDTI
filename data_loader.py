import numpy as np
import pandas as pd
dir_path = 'D:/Users/czx/PycharmProjects/2-server-Experiment/'
feature_path_dr = dir_path + "./feature/drug/"
feature_path_p = dir_path + "./feature/protein/"
sim_feature_path_dr = dir_path + "./feature_sim/drug/"
sim_feature_path_p = dir_path + "./feature_sim/protein/"

def Get_finger():
    MACCS = np.loadtxt(feature_path_dr + "MACCS.csv", dtype=float, delimiter=",")
    Pubchem = np.loadtxt(feature_path_dr + "Pubchem.csv", dtype=float, delimiter=",")
    RDK = np.loadtxt(feature_path_dr + "RDK.csv", dtype=float, delimiter=",")
    ECFP4 = np.loadtxt(feature_path_dr + "ECFP4.csv", dtype=float, delimiter=",")
    FCFP4 = np.loadtxt(feature_path_dr + "FCFP4.csv", dtype=float, delimiter=",")
    Dr_finger = {'maccs': MACCS, 'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
    return Dr_finger

def Get_seq():
    TPC = np.loadtxt(feature_path_p + "TPC.csv", dtype=float, delimiter=",", skiprows=0)
    PAAC = np.loadtxt(feature_path_p + "PAAC.csv", dtype=float, delimiter=",", skiprows=0)
    KSCTriad = np.loadtxt(feature_path_p + "KSCTriad.csv", dtype=float, delimiter=",", skiprows=0)
    CKSAAP = np.loadtxt(feature_path_p + "CKSAAP.csv", dtype=float, delimiter=",", skiprows=0)
    CTD = np.loadtxt(feature_path_p + "CTD.csv", dtype=float, delimiter=",", skiprows=0)
    P_seq = {'PAAC': PAAC, 'KSCTriad': KSCTriad, 'TPC': TPC, 'CKSAAP': CKSAAP, 'CTD': CTD}
    return P_seq

feature_path_dr_pca = 'D:/Users/czx/PycharmProjects/2-Experiment/PCA_feature/drug/'
feature_path_p_pca = 'D:/Users/czx/PycharmProjects/2-Experiment/PCA_feature/protein/'

def Get_PCA_finger():
    MACCS = np.loadtxt(feature_path_dr_pca + "MACCS.csv", dtype=float, delimiter=",")
    Pubchem = np.loadtxt(feature_path_dr_pca + "Pubchem.csv", dtype=float, delimiter=",")
    RDK = np.loadtxt(feature_path_dr_pca + "RDK.csv", dtype=float, delimiter=",")
    ECFP4 = np.loadtxt(feature_path_dr_pca + "ECFP4.csv", dtype=float, delimiter=",")
    FCFP4 = np.loadtxt(feature_path_dr_pca + "FCFP4.csv", dtype=float, delimiter=",")
    Dr_finger = {'maccs': MACCS, 'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
    return Dr_finger

def Get_PCA_seq():
    TPC = np.loadtxt(feature_path_p_pca + "TPC.csv", dtype=float, delimiter=",", skiprows=0)
    PAAC = np.loadtxt(feature_path_p + "PAAC.csv", dtype=float, delimiter=",", skiprows=0)
    KSCTriad = np.loadtxt(feature_path_p_pca + "KSCTriad.csv", dtype=float, delimiter=",", skiprows=0)
    CKSAAP = np.loadtxt(feature_path_p_pca + "CKSAAP.csv", dtype=float, delimiter=",", skiprows=0)
    CTD = np.loadtxt(feature_path_p_pca + "CTD.csv", dtype=float, delimiter=",", skiprows=0)
    P_seq = {'PAAC': PAAC, 'KSCTriad': KSCTriad, 'TPC': TPC, 'CKSAAP': CKSAAP, 'CTD': CTD}
    return P_seq

feature_path_dr_nmf = 'D:/Users/czx/PycharmProjects/2-Experiment/NMF_feature/drug/'
feature_path_p_nmf = 'D:/Users/czx/PycharmProjects/2-Experiment/NMF_feature/protein/'

def Get_NMF_finger():
    MACCS = np.loadtxt(feature_path_dr_nmf + "MACCS.csv", dtype=float, delimiter=",")
    Pubchem = np.loadtxt(feature_path_dr_nmf + "Pubchem.csv", dtype=float, delimiter=",")
    RDK = np.loadtxt(feature_path_dr_nmf + "RDK.csv", dtype=float, delimiter=",")
    ECFP4 = np.loadtxt(feature_path_dr_nmf + "ECFP4.csv", dtype=float, delimiter=",")
    FCFP4 = np.loadtxt(feature_path_dr_nmf + "FCFP4.csv", dtype=float, delimiter=",")
    Dr_finger = {'maccs': MACCS, 'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
    return Dr_finger

def Get_NMF_seq():
    TPC = np.loadtxt(feature_path_p_nmf + "TPC.csv", dtype=float, delimiter=",", skiprows=0)
    PAAC = np.loadtxt(feature_path_p + "PAAC.csv", dtype=float, delimiter=",", skiprows=0)
    KSCTriad = np.loadtxt(feature_path_p_nmf + "KSCTriad.csv", dtype=float, delimiter=",", skiprows=0)
    CKSAAP = np.loadtxt(feature_path_p_nmf + "CKSAAP.csv", dtype=float, delimiter=",", skiprows=0)
    CTD = np.loadtxt(feature_path_p_nmf + "CTD.csv", dtype=float, delimiter=",", skiprows=0)
    P_seq = {'PAAC': PAAC, 'KSCTriad': KSCTriad, 'TPC': TPC, 'CKSAAP': CKSAAP, 'CTD': CTD}
    return P_seq


def Get_drug_sim():
    MACCS = np.loadtxt(sim_feature_path_dr + "MACCS.csv", dtype=float, delimiter=",", skiprows=1)
    Pubchem = np.loadtxt(sim_feature_path_dr + "Pubchem.csv", dtype=float, delimiter=",", skiprows=1)
    RDK = np.loadtxt(sim_feature_path_dr + "RDK.csv", dtype=float, delimiter=",", skiprows=1)
    ECFP4 = np.loadtxt(sim_feature_path_dr + "ECFP4.csv", dtype=float, delimiter=",", skiprows=1)
    FCFP4 = np.loadtxt(sim_feature_path_dr + "FCFP4.csv", dtype=float, delimiter=",", skiprows=1)
    drug_DDI_sim = np.loadtxt(sim_feature_path_dr + "drug_DDI_sim.csv", dtype=float, delimiter=",", skiprows=1)
    drug_Dr_D_sim = np.loadtxt(sim_feature_path_dr + "drug_Dr_D_sim.csv", dtype=float, delimiter=",", skiprows=1)
    Dr_sim = {'maccs': MACCS, 'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4, 'DDI':drug_DDI_sim, 'Dr_D':drug_Dr_D_sim}
    return Dr_sim

def Get_protein_sim():
    seq_sim = np.loadtxt(sim_feature_path_p + "P_seqsim_1771.csv", dtype=float, delimiter=",", skiprows=1)
    PPI_sim = np.loadtxt(sim_feature_path_p + "protein_PPI_sim.csv", dtype=float, delimiter=",", skiprows=1)
    PPI2_sim = np.loadtxt(sim_feature_path_p + "protein_PPI_sim2.csv", dtype=float, delimiter=",", skiprows=1)
    P_D_sim = np.loadtxt(sim_feature_path_p + "protein_P_D_sim.csv", dtype=float, delimiter=",", skiprows=1)
    MF_sim = np.loadtxt(sim_feature_path_p + "GO_sim/protein_sim_MF.csv", dtype=float, delimiter=",", skiprows=1)
    BP_sim = np.loadtxt(sim_feature_path_p + "GO_sim/protein_sim_BP.csv", dtype=float, delimiter=",", skiprows=1)
    CC_sim = np.loadtxt(sim_feature_path_p + "GO_sim/protein_sim_CC.csv", dtype=float, delimiter=",", skiprows=1)
    P_sim = {'seq': seq_sim, 'PPI': PPI_sim, 'PPI2': PPI2_sim, 'P_D': P_D_sim, 'MF': MF_sim, 'BP': BP_sim, 'CC': CC_sim}
    return P_sim

def Get_id():
    Drug_structure = pd.read_csv("./origin_data/drug_structure_1520.csv", sep=',', dtype=str)
    Protein_seq = pd.read_csv(r"./origin_data/protein_seq_1771.csv", sep=',', dtype=str)
    Drug_id, Protein_id = [], []
    for i in range(len(Drug_structure)):
        Drug_id.append(Drug_structure['drugbank'][i])
    for j in range(len(Protein_seq)):
        Protein_id.append(Protein_seq['uniprot'][j])
    return Drug_id, Protein_id

def Get_String():
    Drug_structure = pd.read_csv("./origin_data/drug_structure_1520.csv", sep=',', dtype=str)
    Protein_seq = pd.read_csv(r"./origin_data/protein_seq_1771.csv", sep=',', dtype=str)
    SMILES, Seq = [], []
    for i in range(len(Drug_structure)):
        SMILES.append(Drug_structure['SMILES'][i])
    for j in range(len(Protein_seq)):
        Seq.append(Protein_seq['seq'][j])
    return SMILES, Seq

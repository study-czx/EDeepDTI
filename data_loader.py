import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import funcs
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Get_feature_numbers(data_type, input_type):
    n_dr_feats, n_p_feats = 0, 0
    if input_type == 'd':
        n_dr_feats, n_p_feats = 5, 5
    elif input_type == 'e':
        n_dr_feats, n_p_feats = 8, 8
    elif input_type == 's':
        if data_type == 'DTI':
            n_dr_feats, n_p_feats = 6, 6
        elif data_type == 'CPI':
            n_dr_feats, n_p_feats = 8, 6
        else:
            n_dr_feats, n_p_feats = 5, 6
    print('number of drug feature types: ', n_dr_feats)
    print('number of protein feature types: ', n_p_feats)
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
    RDK = np.loadtxt(feature_path_dr + 'RDKit.csv', dtype=float, delimiter=',', skiprows=1)
    ECFP4 = np.loadtxt(feature_path_dr + 'ECFP4.csv', dtype=float, delimiter=',', skiprows=1)
    FCFP4 = np.loadtxt(feature_path_dr + 'FCFP4.csv', dtype=float, delimiter=',', skiprows=1)
    Dr_finger = {'maccs': MACCS, 'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
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
    RDK = np.loadtxt(sim_feature_path_dr + 'RDKit.csv', dtype=float, delimiter=',')
    ECFP4 = np.loadtxt(sim_feature_path_dr + 'ECFP4.csv', dtype=float, delimiter=',')
    FCFP4 = np.loadtxt(sim_feature_path_dr + 'FCFP4.csv', dtype=float, delimiter=',')
    if data_type == 'DTI':
        DDI_sim = np.loadtxt(sim_feature_path_dr + 'DDI.csv', dtype=float, delimiter=',')
        Dr_sim = {'maccs': MACCS, 'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4, 'DDI': DDI_sim}
        return Dr_sim
    else:
        Dr_sim = {'maccs': MACCS, 'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
        return Dr_sim


def Get_protein_sim(data_type):
    _, _, _, sim_feature_path_p, _, _ = Get_data_path(data_type)
    seq_sim = np.loadtxt(sim_feature_path_p + 'seq.csv', dtype=float, delimiter=',', skiprows=1)
    PPI_sim = np.loadtxt(sim_feature_path_p + 'PPI_a.csv', dtype=float, delimiter=',', skiprows=0)
    PPI2_sim = np.loadtxt(sim_feature_path_p + 'PPI_t.csv', dtype=float, delimiter=',', skiprows=0)
    MF_sim = np.loadtxt(sim_feature_path_p + 'MF.csv', dtype=float, delimiter=',', skiprows=1)
    BP_sim = np.loadtxt(sim_feature_path_p + 'BP.csv', dtype=float, delimiter=',', skiprows=1)
    CC_sim = np.loadtxt(sim_feature_path_p + 'CC.csv', dtype=float, delimiter=',', skiprows=1)
    P_sim = {'seq': seq_sim, 'PPI': PPI_sim, 'PPI2': PPI2_sim, 'MF': MF_sim, 'BP': BP_sim, 'CC': CC_sim}
    return P_sim


def Get_drug_embedding(data_type):
    _, _, _, _, emb_feature_path_dr, _ = Get_data_path(data_type)
    chemberta = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_emb.csv', dtype=float, delimiter=',')
    grover_atom = np.loadtxt(emb_feature_path_dr + 'grover_atom.csv', dtype=float, delimiter=',')
    grover_bond = np.loadtxt(emb_feature_path_dr + 'grover_bond.csv', dtype=float, delimiter=',')
    molformer = np.loadtxt(emb_feature_path_dr + 'Molformer_emb.csv', dtype=float, delimiter=',')
    # molclr = np.loadtxt(emb_feature_path_dr + 'molclr_emb.csv', dtype=float, delimiter=',')
    chemberta_max = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_emb_max.csv', dtype=float, delimiter=',')
    grover_atom_max = np.loadtxt(emb_feature_path_dr + 'grover_atom_max.csv', dtype=float, delimiter=',')
    grover_bond_max = np.loadtxt(emb_feature_path_dr + 'grover_bond_max.csv', dtype=float, delimiter=',')
    molformer_max = np.loadtxt(emb_feature_path_dr + 'Molformer_emb_max.csv', dtype=float, delimiter=',')
    # molclr_max = np.loadtxt(emb_feature_path_dr + 'molclr_emb_max.csv', dtype=float, delimiter=',')
    Dr_embedding = {'chemberta': chemberta, 'grover_atom': grover_atom, 'grover_bond': grover_bond,
                    'molformer': molformer, 'chemberta_max': chemberta_max, 'grover_atom_max': grover_atom_max,
                    'grover_bond_max': grover_bond_max, 'molformer_max': molformer_max}
    return Dr_embedding


def Get_protein_embedding(data_type):
    _, _, _, _, _, emb_feature_path_p = Get_data_path(data_type)
    esm2 = np.loadtxt(emb_feature_path_p + 'ESM2_emb.csv', dtype=float, delimiter=',')
    protein_bert = np.loadtxt(emb_feature_path_p + 'Protein_bert_emb.csv', dtype=float, delimiter=',')
    prottrans = np.loadtxt(emb_feature_path_p + 'ProtTrans_emb.csv', dtype=float, delimiter=',')
    tape = np.loadtxt(emb_feature_path_p + 'TAPE_emb.csv', dtype=float, delimiter=',')
    esm2_max = np.loadtxt(emb_feature_path_p + 'ESM2_emb_max.csv', dtype=float, delimiter=',')
    protein_bert_max = np.loadtxt(emb_feature_path_p + 'Protein_bert_emb_max.csv', dtype=float, delimiter=',')
    prottrans_max = np.loadtxt(emb_feature_path_p + 'ProtTrans_emb_max.csv', dtype=float, delimiter=',')
    tape_max = np.loadtxt(emb_feature_path_p + 'TAPE_emb_max.csv', dtype=float, delimiter=',')
    P_embedding = {'esm2': esm2, 'protein_bert': protein_bert, 'prottrans': prottrans, 'tape': tape,
                   'esm2_max': esm2_max, 'protein_bert_max': protein_bert_max, 'prottrans_max': prottrans_max,
                   'tape_max': tape_max}
    return P_embedding


def Get_id(data_type):
    if data_type == 'DTI':
        Drug_id = pd.read_csv('datasets_DTI/datasets/DTI/Drug_id.csv', sep=',', dtype=str)
        Protein_id = pd.read_csv('datasets_DTI/datasets/DTI/Protein_id.csv', sep=',', dtype=str)
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


def Trans_feature(Feature):
    for i in Feature:
        Feature[i] = torch.as_tensor(torch.from_numpy(Feature[i]), dtype=torch.float32).to(device)
    return Feature


def Get_feature(data_type, input_type):
    Drug_id, Protein_id = Get_id(data_type)
    n_drugs, n_proteins = len(Drug_id), len(Protein_id)
    dr_id_map, p_id_map = funcs.id_map(Drug_id), funcs.id_map(Protein_id)
    # print("number of Drug: ", n_drugs)
    # print("number of Protein ", n_proteins)

    if input_type == 'd':
        Dr_finger, P_seq = Get_finger(data_type), Get_seq(data_type)
        Dr_finger, P_seq = Trans_feature(Dr_finger), Trans_feature(P_seq)
        Drug_features = [Dr_finger['ecfp4'], Dr_finger['fcfp4'], Dr_finger['pubchem'], Dr_finger['maccs'],
                         Dr_finger['rdk']]
        Protein_features = [P_seq['KSCTriad'], P_seq['CKSAAP'], P_seq['TPC'], P_seq['PAAC'], P_seq['CTD']]
        return dr_id_map, p_id_map, Drug_features, Protein_features
    elif input_type == 'e':
        Dr_embedding, P_embedding = Get_drug_embedding(data_type), Get_protein_embedding(data_type)
        Dr_embedding, P_embedding = Trans_feature(Dr_embedding), Trans_feature(P_embedding)
        Drug_features = [Dr_embedding['chemberta'], Dr_embedding['grover_atom'], Dr_embedding['grover_bond'],
                         Dr_embedding['molformer'], Dr_embedding['chemberta_max'],
                         Dr_embedding['grover_atom_max'], Dr_embedding['grover_bond_max'],
                         Dr_embedding['molformer_max']]
        Protein_features = [P_embedding['esm2'], P_embedding['protein_bert'], P_embedding['prottrans'],
                            P_embedding['tape'], P_embedding['esm2_max'], P_embedding['protein_bert_max'],
                            P_embedding['prottrans_max'], P_embedding['tape_max']]
        return dr_id_map, p_id_map, Drug_features, Protein_features
    elif input_type == 's':
        if data_type == 'DTI':
            Dr_sim, P_sim = Get_drug_sim(data_type), Get_protein_sim(data_type)
            Dr_sim, P_sim = Trans_feature(Dr_sim), Trans_feature(P_sim)
            Drug_features = [Dr_sim['ecfp4'], Dr_sim['fcfp4'], Dr_sim['pubchem'], Dr_sim['maccs'],
                             Dr_sim['rdk'], Dr_sim['DDI']]
            Protein_features = [P_sim['seq'], P_sim['MF'], P_sim['BP'], P_sim['CC'],
                                P_sim['PPI'], P_sim['PPI2']]
            return dr_id_map, p_id_map, Drug_features, Protein_features
        elif data_type == 'CPI':
            Dr_finger, P_sim = Get_finger(data_type), Get_protein_sim(data_type)
            Dr_finger, P_sim = Trans_feature(Dr_finger), Trans_feature(P_sim)
            Drug_features = [Dr_finger['ecfp4'], Dr_finger['fcfp4'], Dr_finger['pubchem'], Dr_finger['maccs'],
                             Dr_finger['rdk']]
            Protein_features = [P_sim['seq'], P_sim['MF'], P_sim['BP'], P_sim['CC'],
                                P_sim['PPI'], P_sim['PPI2']]
            return dr_id_map, p_id_map, Drug_features, Protein_features
        else:
            Dr_sim, P_sim = Get_drug_sim(data_type), Get_protein_sim(data_type)
            Dr_sim, P_sim = Trans_feature(Dr_sim), Trans_feature(P_sim)
            Drug_features = [Dr_sim['ecfp4'], Dr_sim['fcfp4'], Dr_sim['pubchem'], Dr_sim['maccs'],
                             Dr_sim['rdk']]
            Protein_features = [P_sim['seq'], P_sim['MF'], P_sim['BP'], P_sim['CC'],
                                P_sim['PPI'], P_sim['PPI2']]
            return dr_id_map, p_id_map, Drug_features, Protein_features


def Get_CPI_feature(data_type, input_type, dr_number, p_number):
    feature_path_dr, feature_path_p, sim_feature_path_dr, sim_feature_path_p, emb_path_dr, emb_path_p = Get_data_path(
        data_type)
    if input_type == 'd':
        drug_feature_names = ['ECFP4', 'FCFP4', 'PubChem', 'MACCS', 'RDKit']
        protein_feature_names = ['KSCTriad', 'CKSAAP', 'TPC', 'PAAC', 'CTD']
        this_dr_name, this_p_name = drug_feature_names[dr_number], protein_feature_names[p_number]
        this_drug_feature = np.loadtxt(feature_path_dr + this_dr_name + '.csv', dtype=float, delimiter=',', skiprows=1)
        this_drug_feature = torch.as_tensor(torch.from_numpy(this_drug_feature), dtype=torch.float32).to(device)
        if this_p_name == 'CTD':
            scaler = MinMaxScaler()
            CTDC = np.loadtxt(feature_path_p + 'CTDC.csv', dtype=float, delimiter=',')
            CTDT = np.loadtxt(feature_path_p + 'CTDT.csv', dtype=float, delimiter=',')
            CTDD = np.loadtxt(feature_path_p + 'CTDD.csv', dtype=float, delimiter=',')
            CTDD = scaler.fit_transform(CTDD)
            this_protein_feature = np.concatenate((CTDC, CTDT, CTDD), axis=1)
            this_protein_feature = torch.as_tensor(torch.from_numpy(this_protein_feature), dtype=torch.float32).to(
                device)
            return this_drug_feature, this_protein_feature
        else:
            this_protein_feature = np.loadtxt(feature_path_p + this_p_name + '.csv', dtype=float, delimiter=',')
            this_protein_feature = torch.as_tensor(torch.from_numpy(this_protein_feature), dtype=torch.float32).to(
                device)
            return this_drug_feature, this_protein_feature
    elif input_type == 'e':
        drug_feature_names = ['ChemBERTa2_emb', 'grover_atom', 'grover_bond', 'Molformer_emb', 'ChemBERTa2_emb_max',
                              'grover_atom_max', 'grover_bond_max', 'Molformer_emb_max']
        protein_feature_names = ['ESM2_emb', 'Protein_bert_emb', 'ProtTrans_emb', 'TAPE_emb', 'ESM2_emb_max',
                                 'Protein_bert_emb_max', 'ProtTrans_emb_max', 'TAPE_emb_max']
        this_dr_name, this_p_name = drug_feature_names[dr_number], protein_feature_names[p_number]
        this_drug_feature = np.loadtxt(emb_path_dr + this_dr_name + '.csv', dtype=float, delimiter=',')
        this_protein_feature = np.loadtxt(emb_path_p + this_p_name + '.csv', dtype=float, delimiter=',')
        this_drug_feature = torch.as_tensor(torch.from_numpy(this_drug_feature), dtype=torch.float32).to(device)
        this_protein_feature = torch.as_tensor(torch.from_numpy(this_protein_feature), dtype=torch.float32).to(device)
        return this_drug_feature, this_protein_feature


    elif input_type == 's':
        drug_feature_names = ['ChemBERTa2_emb', 'grover_atom', 'grover_bond', 'Molformer_emb', 'ChemBERTa2_emb_max',
                              'grover_atom_max', 'grover_bond_max', 'Molformer_emb_max']
        protein_feature_names = ['seq', 'PPI_a', 'PPI_t', 'MF', 'BP', 'CC']
        this_dr_name, this_p_name = drug_feature_names[dr_number], protein_feature_names[p_number]
        this_drug_feature = np.loadtxt(emb_path_dr + this_dr_name + '.csv', dtype=float, delimiter=',')
        this_drug_feature = torch.as_tensor(torch.from_numpy(this_drug_feature), dtype=torch.float32).to(device)

        if this_p_name == 'PPI_a' or this_p_name == 'PPI_t':
            this_protein_feature = np.loadtxt(sim_feature_path_p + this_p_name + '.csv', dtype=float, delimiter=',',
                                              skiprows=0)
            this_protein_feature = torch.as_tensor(torch.from_numpy(this_protein_feature), dtype=torch.float32).to(
                device)
        else:
            this_protein_feature = np.loadtxt(sim_feature_path_p + this_p_name + '.csv', dtype=float, delimiter=',',
                                              skiprows=1)
            this_protein_feature = torch.as_tensor(torch.from_numpy(this_protein_feature), dtype=torch.float32).to(
                device)
        return this_drug_feature, this_protein_feature


# Davis and KIBA : DK
def Get_DK_finger(dataset):
    path_dr = dataset + 'drug_feature/'
    MACCS = np.loadtxt(path_dr + 'MACCS.csv', dtype=float, delimiter=',')
    Pubchem = np.loadtxt(path_dr + 'Pubchem.csv', dtype=float, delimiter=',')
    RDK = np.loadtxt(path_dr + 'RDKit.csv', dtype=float, delimiter=',')
    ECFP4 = np.loadtxt(path_dr + 'ECFP4_2048.csv', dtype=float, delimiter=',')
    FCFP4 = np.loadtxt(path_dr + 'FCFP4_2048.csv', dtype=float, delimiter=',')
    Dr_finger = {'maccs': MACCS, 'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
    return Dr_finger


def Get_DK_seq(dataset):
    path_p = dataset + 'protein_feature/'
    TPC = np.loadtxt(path_p + 'TPC.csv', dtype=float, delimiter=',', skiprows=0)
    PAAC = np.loadtxt(path_p + 'PAAC.csv', dtype=float, delimiter=',', skiprows=0)
    KSCTriad = np.loadtxt(path_p + 'KSCTriad.csv', dtype=float, delimiter=',', skiprows=0)
    CKSAAP = np.loadtxt(path_p + 'CKSAAP.csv', dtype=float, delimiter=',', skiprows=0)
    CTD = np.loadtxt(path_p + 'CTD.csv', dtype=float, delimiter=',', skiprows=0)
    P_seq = {'PAAC': PAAC, 'KSCTriad': KSCTriad, 'TPC': TPC, 'CKSAAP': CKSAAP, 'CTD': CTD}
    return P_seq


def Get_DK_String(dataset):
    Drug_structure = pd.read_csv(dataset + 'Drug.csv', sep=',', dtype=str)
    Protein_seq = pd.read_csv(dataset + 'Protein.csv', sep=',', dtype=str)
    SMILES, Seq = [], []
    for i in range(len(Drug_structure)):
        SMILES.append(Drug_structure['smiles'][i])
    for j in range(len(Protein_seq)):
        Seq.append(Protein_seq['seq'][j])
    return SMILES, Seq


def Get_DK_id(dataset):
    Drug_structure = pd.read_csv(dataset + 'Drug.csv', sep=',', dtype=str)
    Protein_seq = pd.read_csv(dataset + 'Protein.csv', sep=',', dtype=str)
    Drug_id, Protein_id = [], []
    for i in range(len(Drug_structure)):
        Drug_id.append(Drug_structure['drug'][i])
    for j in range(len(Protein_seq)):
        Protein_id.append(Protein_seq['protein'][j])
    return Drug_id, Protein_id


def Get_DK_drug_sim(dataset):
    path_dr = dataset + 'drug_feature_sim/'
    MACCS = np.loadtxt(path_dr + 'MACCS.csv', dtype=float, delimiter=',', skiprows=1)
    Pubchem = np.loadtxt(path_dr + 'Pubchem.csv', dtype=float, delimiter=',', skiprows=1)
    RDK = np.loadtxt(path_dr + 'RDKit.csv', dtype=float, delimiter=',', skiprows=1)
    ECFP4 = np.loadtxt(path_dr + 'ECFP4.csv', dtype=float, delimiter=',', skiprows=1)
    FCFP4 = np.loadtxt(path_dr + 'FCFP4.csv', dtype=float, delimiter=',', skiprows=1)
    Dr_sim = {'maccs': MACCS, 'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
    return Dr_sim


def Get_DK_protein_sim(dataset):
    path_p = dataset + 'protein_feature_sim/'
    seq_sim = np.loadtxt(path_p + 'seq.csv', dtype=float, delimiter=',', skiprows=0)
    PPI_sim = np.loadtxt(path_p + 'PPI_sim.csv', dtype=float, delimiter=',', skiprows=0)
    PPI2_sim = np.loadtxt(path_p + 'PPI2_sim.csv', dtype=float, delimiter=',', skiprows=0)
    MF_sim = np.loadtxt(path_p + 'MF_sim.csv', dtype=float, delimiter=',', skiprows=0)
    BP_sim = np.loadtxt(path_p + 'BP_sim.csv', dtype=float, delimiter=',', skiprows=0)
    CC_sim = np.loadtxt(path_p + 'CC_sim.csv', dtype=float, delimiter=',', skiprows=0)
    P_sim = {'seq': seq_sim, 'PPI': PPI_sim, 'PPI2': PPI2_sim, 'MF': MF_sim, 'BP': BP_sim, 'CC': CC_sim}
    return P_sim

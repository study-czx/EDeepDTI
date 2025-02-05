import os

import pandas as pd
import numpy as np
from multiprocessing import Pool
import dgl.backend as F
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem
from scipy import sparse as sp
from src.model_config import config_dict

from src.data.featurizer import smiles_to_graph_tune
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
from src.utils import set_random_seed
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import MoleculeDataset, MoleculeDataset_CPI
from src.data.collator import Collator_tune
from src.model.light import LiGhTPredictor as LiGhT
import warnings
import math
import gc

warnings.filterwarnings("ignore")

path_length = 5
n_jobs = 32

set_random_seed(22,1)




def preprocess_dataset(base_path, df, dataset):
    cache_file_path = base_path + dataset + '_' + str(path_length) + ".pkl"
    smiless = df.smiles.values.tolist()
    task_names = df.columns.drop(['smiles']).tolist()
    print('constructing graphs')

    graphs = pmap(smiles_to_graph_tune,
                  smiless,
                  max_length=path_length,
                  n_virtual_nodes=2,
                  n_jobs=n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]
    print('saving graphs')
    save_graphs(cache_file_path, valid_graphs, labels={'labels': labels})

    print('extracting fingerprints')
    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')
    sp.save_npz(base_path + "rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(n_jobs).imap(generator.process, smiless)
    # features_map = [generator.process(smiles) for smiles in smiless]
    arr = np.array(list(features_map))
    np.savez_compressed(base_path + "/molecular_descriptors.npz", md=arr[:, 1:])
    del features_map, FP_sp_mat, FP_arr, FP_list, valid_graphs, graphs

    gc.collect()

def extract_features(config_name, model_path, data_path, dataset, this_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)

    config = config_dict['base']
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size
    ).to(device)
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('models/pretrained/base/base.pth').items()})
    fps_list = []
    collator = Collator_tune(config['path_length'])
    mol_dataset = MoleculeDataset_CPI(df=this_df, root_path=data_path, dataset = dataset, dataset_type=None)
    loader = DataLoader(mol_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False, collate_fn=collator)

    for batch_idx, batched_data in enumerate(loader):
        (_, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(device)
        md = md.to(device)
        g = g.to(device)
        fps = model.generate_fps(g, ecfp, md)
        fps_list.extend(fps.detach().cpu().numpy().tolist())
    # print(np.array(fps_list).shape)
    # kpgt_emb = pd.DataFrame(fps_list)
    # kpgt_emb.to_csv(f"{data_path}/{dataset}/kpgt_emb.csv", index=False, header=False)
    return fps_list


if __name__ == "__main__":
    types = ['CPI']
    path_name_dict = {'DTI': 'drug_smiles', 'CPI': 'all_compound_smiles', 'Davis': 'Drug', 'KIBA': 'Drug'}
    smiles_name_dict = {'DTI': 'SMILES', 'CPI': 'CanonicalSMILES', 'Davis': 'smiles', 'KIBA': 'smiles'}

    for type in types:
        base_path = 'data/' + type + '/'
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        smiles_list = pd.read_csv(base_path + type + '.csv')

        my_CPI_batch = 10000
        all_kgpt_list = []
        for i in range(math.ceil(len(smiles_list)/my_CPI_batch)):
            start = i * my_CPI_batch
            end = start + my_CPI_batch
            print(start, end)
            batch_smiles = smiles_list.iloc[start:min(end, len(smiles_list)),:]
            print(batch_smiles)
            # smiles_name = smiles_name_dict[type]
            smiles = batch_smiles[['smiles']]
            # smiles.columns = ['smiles']
            preprocess_dataset(base_path, smiles, type)

            kgpt_emb = extract_features('base', 'models/pretrained/base/base.pth', 'data/', type, smiles)
            kgpt_emb = pd.DataFrame(kgpt_emb)
            kgpt_emb.to_csv(f"data/"+type+"/kpgt_emb.csv", mode='a', index=False, header=False)
            # kgpt_emb.to_csv(f"data/" + type + "/kpgt_emb_max.csv", mode='a', index=False, header=False)
            print('write OK')
            # all_kgpt_list.extend(kgpt_emb)
            del kgpt_emb
            gc.collect()

        # all_kgpt_emb = pd.DataFrame(all_kgpt_list)
        # all_kgpt_emb.to_csv(f"data/"+type+"/kpgt_emb.csv", mode='a', index=False, header=False)
        # all_kgpt_emb.to_csv(f"data/"+type+"/kpgt_emb_max.csv", index=False, header=False)


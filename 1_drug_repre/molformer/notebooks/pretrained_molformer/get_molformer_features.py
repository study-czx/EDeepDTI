from argparse import Namespace
import yaml
from tokenizer.tokenizer import MolTranBertTokenizer
from train_pubchem_light import LightningModule
import pandas as pd
from rdkit import Chem
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('../../data/Pretrained MoLFormer/hparams.yaml', 'r') as f:
    config = Namespace(**yaml.safe_load(f))

tokenizer = MolTranBertTokenizer('bert_vocab.txt')
# tokenizer.vocab
ckpt = '../../data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab).to(device)
# lm

import torch
from fast_transformers.masking import LengthMask as LM

def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size

def canonicalize(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)


def embed(model, smiles, tokenizer, batch_size=640):
    model.eval()
    embeddings = []
    max_embeddings = []
    for batch in batch_split(smiles, batch_size=batch_size):
        print('one batch')
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']).to(device), torch.tensor(batch_enc['attention_mask']).to(device)
        with torch.no_grad():
            token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
        # average pooling over tokens
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        need_embeddings = token_embeddings * input_mask_expanded
        # print(need_embeddings.shape)
        sum_embeddings = torch.sum(need_embeddings, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        # print(embedding.shape)
        embedding_max = need_embeddings.max(dim=1)[0].cpu().detach()
        # print(embedding_max.shape)
        embeddings.append(embedding.detach().cpu())
        max_embeddings.append(embedding_max)
    return torch.cat(embeddings).numpy(), torch.cat(max_embeddings).numpy()


types = ['Davis', 'KIBA', 'DTI', 'CPI']
path_name_dict = {'DTI': 'drug_smiles', 'CPI': 'all_compound_smiles', 'Davis': 'Drug', 'KIBA': 'Drug'}
smiles_name_dict = {'DTI': 'SMILES', 'CPI': 'CanonicalSMILES', 'Davis': 'smiles', 'KIBA': 'smiles'}

for type in types:
    # prepare your protein sequence as a list
    smiles_list = pd.read_csv('../../data/' + type + '/' + path_name_dict[type] + '.csv')
    smiles_name = smiles_name_dict[type]
    smiles = smiles_list[smiles_name].apply(canonicalize)

    X, X_max = embed(lm, smiles, tokenizer)
    print(X.shape)
    print(X_max.shape)
    all_emb_df = pd.DataFrame(X)
    all_emb_df_max = pd.DataFrame(X_max)
    # all_emb_df2 = pd.DataFrame(smiles_representations_max)
    output_path1 = '../../data/' + type + '/Molformer_emb.csv'
    output_path2 = '../../data/' + type + '/Molformer_emb_max.csv'
    all_emb_df.to_csv(output_path1, index=False, header=False)
    all_emb_df_max.to_csv(output_path2, index=False, header=False)


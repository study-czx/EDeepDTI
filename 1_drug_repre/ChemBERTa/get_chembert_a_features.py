from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_types = ['MTR', 'MLM']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for model_type in model_types:
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-"+model_type)
    model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-"+model_type, output_hidden_states=True).to(device)
    model.eval()

    types = ['DTI', 'Davis', 'KIBA', 'CPI']
    path_name_dict = {'DTI': 'drug_smiles', 'CPI': 'all_compound_smiles', 'Davis': 'Drug', 'KIBA': 'Drug'}
    smiles_name_dict = {'DTI': 'SMILES', 'CPI': 'CanonicalSMILES', 'Davis': 'smiles', 'KIBA': 'smiles'}

    for type in types:
        # prepare your protein sequence as a list
        drug_smiles_list = pd.read_csv('data/' + type + '/' + path_name_dict[type] + '.csv')
        smiles_representations = []
        smiles_representations_max = []

        for k in range(len(drug_smiles_list)):
            drug_id = drug_smiles_list.iloc[k, 0]
            print(drug_id)
            smiles = drug_smiles_list.iloc[k, 1]
            len_this_smiles = len(str(smiles))
            max_length = 510
            if len_this_smiles > max_length:
                smiles = str(smiles)[0:max_length]
                print('too long')
                print('length of this smiles trans to: ', len(str(smiles)))
            with torch.no_grad():
                inputs = tokenizer(smiles, return_tensors="pt").to(device)
                # print(inputs)
                outputs = model(**inputs)
                hidden_states = outputs["hidden_states"][-1][0]

                this_representation = hidden_states.mean(0).cpu().detach().numpy()
                this_representation_max = hidden_states.max(0)[0].cpu().detach().numpy()
                smiles_representations.append(this_representation)
                smiles_representations_max.append(this_representation_max)
                # print(outputs)
                # print(this_representation.shape)

        all_emb_df = pd.DataFrame(smiles_representations)
        all_emb_df2 = pd.DataFrame(smiles_representations_max)
        output_path = 'data/' + type + '/ChemBERTa2_emb_'+model_type+'.csv'
        output_path2 = 'data/' + type + '/ChemBERTa2_emb_'+model_type+'_max.csv'
        all_emb_df.to_csv(output_path, index=False, header=False)
        all_emb_df2.to_csv(output_path2, index=False, header=False)
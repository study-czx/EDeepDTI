# -*- coding: utf-8 -*-

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16).to(device)
model.eval()

types = ['DTI', 'CPI', 'Davis', 'KIBA']
# types = ['CPI', 'Davis', 'KIBA']
path_name_dict = {'DTI': 'protein_sequence', 'CPI': 'all_protein_sequence', 'Davis': 'Protein', 'KIBA': 'Protein'}
seq_name_dict = {'DTI': 'Sequence', 'CPI': 'Sequence', 'Davis': 'seq', 'KIBA': 'seq'}
for type in types:
    # prepare your protein sequence as a list
    protein_sequences_list = pd.read_csv('../' + type + '/' + path_name_dict[type] + '.csv')
    sequences = protein_sequences_list[seq_name_dict[type]].tolist()
    print(len(sequences))

    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids

    all_emb1 = []
    all_emb2 = []
    all_number_sequence = len(sequences)
    max_length = 1200

    # tokenize sequence and pad up to the longest sequence in the batch
    with torch.no_grad():
        for i in range(all_number_sequence):
            this_sequence = sequences[i]
            len_this_sequence = len(str(this_sequence))
            print(len(this_sequence))
            if len_this_sequence > max_length:
                this_sequence = str(this_sequence)[0:max_length]
                print('too long')
                print(len(this_sequence))
                len_this_sequence = max_length

            this_sequence = [this_sequence]
            this_sequence = [" ".join(list(re.sub(r"[UZOB]", "X", this_sequence))) for this_sequence in this_sequence]
            ids = tokenizer(this_sequence, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
            emb = embedding_repr.last_hidden_state[0, :len_this_sequence]  # shape (7 x 1024)
            # print(emb.shape)
            emb_0_per_protein_mean = emb.mean(dim=0).cpu().detach().numpy()  # shape (1024)
            emb_0_per_protein_max = emb.max(dim=0)[0].cpu().detach().numpy()  # shape (1024)
            # print(emb_0_per_protein.shape)
            # print(emb_0_per_protein)
            all_emb1.append(emb_0_per_protein_mean)
            all_emb2.append(emb_0_per_protein_max)

    all_emb_df1 = pd.DataFrame(all_emb1)
    all_emb_df2 = pd.DataFrame(all_emb2)
    output_path1 = '../' + type + '/ProtTrans_emb.csv'
    output_path2 = '../' + type + '/ProtTrans_emb_max.csv'
    all_emb_df1.to_csv(output_path1, index=False, header=False)
    all_emb_df2.to_csv(output_path2, index=False, header=False)

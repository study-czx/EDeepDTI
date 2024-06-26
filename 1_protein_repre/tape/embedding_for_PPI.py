import torch
from tape import ProteinBertModel, TAPETokenizer
import pandas as pd


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ProteinBertModel.from_pretrained('bert-base')
model = model.to(device)

tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model

types = ['S.cerevisiae']


for type in types:
    # prepare your protein sequence as a list
    protein_sequences_list = pd.read_csv('../' + type + '/seq.csv')
    sequences = protein_sequences_list['sequence'].tolist()

    all_number_sequence = len(sequences)
    max_length = 2400

    all_emb1 = []
    all_emb2 = []
    with torch.no_grad():
        for i in range(all_number_sequence):
            this_sequence = sequences[i]
            # print(this_sequence)
            # print(this_sequence)
            len_this_sequence = len(str(this_sequence))
            # print(len(this_sequence))
            if len_this_sequence > max_length:
                this_sequence = str(this_sequence)[0:max_length]
                print('too long')
                print(len(this_sequence))
                len_this_sequence = max_length

            token_ids = torch.tensor([tokenizer.encode(this_sequence)]).to(device)
            output = model(token_ids)
            sequence_output = output[0][0]
            pooled_output = output[1][0]
            emb_per_protein_mean = sequence_output.mean(dim=0).cpu().detach().numpy()  # shape (768)
            emb_per_protein_max = sequence_output.max(dim=0)[0].cpu().detach().numpy()  # shape (768)
            all_emb1.append(emb_per_protein_mean)
            all_emb2.append(emb_per_protein_max)

    all_emb_df1 = pd.DataFrame(all_emb1)
    all_emb_df2 = pd.DataFrame(all_emb2)
    output_path1 = '../' + type + '/TAPE_emb.csv'
    output_path2 = '../' + type + '/TAPE_emb_max.csv'
    all_emb_df1.to_csv(output_path1, index=False, header=False)
    all_emb_df2.to_csv(output_path2, index=False, header=False)
# NOTE: pooled_output is *not* trained for the transformer, do not use
# w/o fine-tuning. A better option for now is to simply take a mean of
# the sequence output
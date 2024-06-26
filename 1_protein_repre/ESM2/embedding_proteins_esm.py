import torch
import esm
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

max_length = 1200
# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
types = ['Davis', 'KIBA', 'DTI', 'CPI']
path_name_dict = {'DTI': 'protein_sequence', 'CPI': 'all_protein_sequence', 'Davis': 'Protein', 'KIBA': 'Protein'}
seq_name_dict = {'DTI': 'Sequence', 'CPI': 'Sequence', 'Davis': 'seq', 'KIBA': 'seq'}

for type in types:
    # prepare your protein sequence as a list
    protein_sequences_list = pd.read_csv('../' + type + '/' + path_name_dict[type] + '.csv')
    sequences = protein_sequences_list[seq_name_dict[type]].tolist()
    print(len(sequences))
    # protein_sequences_list = protein_sequences_list.iloc[0:4, :]
    sequence_representations = []
    sequence_representations_max = []

    for k in range(len(sequences)):
        protein_id = protein_sequences_list.iloc[k, 0]
        print(protein_id)
        sequence = protein_sequences_list.iloc[k, 1]
        len_this_sequence = len(str(sequence))
        if len_this_sequence > max_length:
            sequence = str(sequence)[0:max_length]
            print('too long')
        this_data = (protein_id, sequence)
        data = [this_data]
        # print(data)

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        # 清除未使用的 GPU 内存
        torch.cuda.empty_cache()

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, tokens_len in enumerate(batch_lens):
            this_representation = token_representations[i, 1: tokens_len - 1].mean(0).cpu().detach().numpy()
            this_representation_max = token_representations[i, 1: tokens_len - 1].max(0)[0].cpu().detach().numpy()
            # print(this_representation)
            # print(this_representation.shape)
            # print(this_representation_max.shape)
            sequence_representations.append(this_representation)
            sequence_representations_max.append(this_representation_max)

        del token_representations, results


    all_emb_df = pd.DataFrame(sequence_representations)
    all_emb_df2 = pd.DataFrame(sequence_representations_max)
    output_path = '../' + type + '/ESM2_emb.csv'
    output_path2 = '../' + type + '/ESM2_emb_max.csv'
    all_emb_df.to_csv(output_path, index=False, header=False)
    all_emb_df2.to_csv(output_path2, index=False, header=False)








from protein_bert.proteinbert import load_pretrained_model
from protein_bert.proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

warnings.filterwarnings('ignore')

types = ['S.cerevisiae']

pretrained_model_generator, input_encoder = load_pretrained_model()

for type in types:
    # prepare your protein sequence as a list
    protein_sequences_list = pd.read_csv('../' + type + '/seq.csv')
    sequences = protein_sequences_list['sequence'].tolist()

    sequence_representations = []
    sequence_representations_max = []

    max_length = 1200
    for k in range(len(protein_sequences_list)):
        protein_id = str(k)
        print(protein_id)
        sequence = protein_sequences_list.iloc[k, 0]
        len_this_sequence = len(str(sequence))
        if len_this_sequence > max_length:
            sequence = str(sequence)[0:max_length]
            print('too long')
            len_this_sequence = max_length
        # print(sequence)

        seq_len = len_this_sequence
        batch_size = 1
        model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len + 2))
        # encode_X function will add START and END token to the two ends of the sequence.
        encoded_x = input_encoder.encode_X([sequence], seq_len)
        local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size)
        # print(local_representations.shape)
        # print(local_representations)
        local_representations_mean = np.mean(np.array(local_representations[0][1:seq_len + 1]), axis=0)
        local_representations_max = np.max(np.array(local_representations[0][1:seq_len + 1]), axis=0)
        # print(local_representations_mean.shape, local_representations_max.shape)
        sequence_representations.append(local_representations_mean)
        sequence_representations_max.append(local_representations_max)

    all_protein_bert_df = pd.DataFrame(sequence_representations)
    all_protein_bert_df2 = pd.DataFrame(sequence_representations_max)
    output_path = '../' + type + '/Protein_bert_emb.csv'
    output_path2 = '../' + type + '/Protein_bert_emb_max.csv'
    all_protein_bert_df.to_csv(output_path, index=False, header=False)
    all_protein_bert_df2.to_csv(output_path2, index=False, header=False)

# ... use these as features for other tasks, based on local_representations, global_representations

import os
import numpy as np
import pandas as pd

types = ['deep']
feature_types = ['atom', 'bond']


for type in types:
    data_path = 'data/DDI_dataset/drug_list_' + type + '.csv'

    # prepare your protein sequence as a list
    os.system('python scripts/save_features.py --data_path ' + data_path + ' \
                                        --save_path data/DDI_dataset/' + type + '/drug_smiles.npz \
                                        --features_generator rdkit_2d_normalized \
                                        --restart ')
    for feature_type in feature_types:
        np_path = 'data/DDI_dataset/' + type + '/fp_' + feature_type + '.npz'
        csv_path = 'data/DDI_dataset/' + type + '_grover_' + feature_type + '.csv'
        # atom, bond, both
        os.system('python main.py fingerprint --data_path ' + data_path + ' \
                                       --features_path data/DDI_dataset/' + type + '/drug_smiles.npz \
                                       --checkpoint_path model/grover_large.pt \
                                       --fingerprint_source ' + feature_type + ' \
                                       --output '+np_path)
    # for feature_type in feature_types:
    #     np_path = 'data/DDI_dataset/' + type + '/fp_' + feature_type + '_max.npz'
    #     csv_path = 'data/DDI_dataset/' + type + '_grover_' + feature_type + '_max.csv'
    #     # atom, bond, both
    #     os.system('python main.py fingerprint --data_path ' + data_path + ' \
    #                                    --features_path data/DDI_dataset/' + type + '/drug_smiles.npz \
    #                                    --checkpoint_path model/grover_large.pt \
    #                                    --fingerprint_source ' + feature_type + ' \
    #                                    --output '+np_path)

        data = np.load(np_path)
        fps = data['fps']
        df = pd.DataFrame(fps)
        print(df.shape)
        df.to_csv(csv_path, index=False, header=False)
        os.system('rm ' + np_path)



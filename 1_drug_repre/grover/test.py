import numpy as np
import pandas as pd

types = ['Davis', 'KIBA', 'DTI', 'CPI']

feature_types = ['atom', 'bond', 'both']
for type in types:
    for feature_type in feature_types:
        np_path = 'data/' + type + '/fp_' + feature_type + '.npz'
        csv_path = 'data/' + type + '/fp_' + feature_type + '.csv'
        data = np.load(np_path)
        fps = data['fps']

        df = pd.DataFrame(fps)
        print(df)
        print(df.shape)
        df.to_csv(csv_path, index=False, header=False)

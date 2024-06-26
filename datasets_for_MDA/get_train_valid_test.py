import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
import random
import numpy as np
from pathlib import Path
import os

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

setup_seed(42)
torch.manual_seed(42)

def Make_path(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        os.makedirs(data_path)

train_pos = pd.read_csv('train_test/train_pos.csv', header=None, dtype=int)
train_neg = pd.read_csv('train_test/train_neg.csv', header=None, dtype=int)

test_pos = pd.read_csv('train_test/test_pos.csv', header=None, dtype=int)
test_neg = pd.read_csv('train_test/test_neg.csv', header=None, dtype=int)

# 将正样本数据和负样本数据合并成一个总的数据集
total_data = np.concatenate((train_pos, train_neg), axis=0)
# 生成相应的标签数组，1表示正样本，0表示负样本
labels = np.concatenate((np.ones(len(train_pos)), np.zeros(len(train_neg))), axis=0)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# kf = KFold(n_splits=5, shuffle=True, random_state=1)
fold_index = 1

for train_index, val_index in kf.split(total_data, labels):
    # 获取当前fold的训练集和验证集数据
    X_train, X_val = total_data[train_index], total_data[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    train_pos = X_train[y_train == 1]
    train_neg = X_train[y_train == 0]
    valid_pos = X_val[y_val == 1]
    valid_neg = X_val[y_val == 0]
    print(len(train_pos), len(valid_pos), len(train_neg), len(valid_neg))
    data_path = 'datasets/fold'+str(fold_index)
    Make_path(data_path)
    pd.DataFrame(train_pos).to_csv(data_path + '/train_pos.csv', index=False)
    pd.DataFrame(train_neg).to_csv(data_path + '/train_neg.csv', index=False)
    pd.DataFrame(valid_pos).to_csv(data_path + '/valid_pos.csv', index=False)
    pd.DataFrame(valid_neg).to_csv(data_path + '/valid_neg.csv', index=False)
    fold_index += 1


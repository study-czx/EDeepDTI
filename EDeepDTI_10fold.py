import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import funcs
import pandas as pd
import data_loader
from model import DNNNet
from sklearn.model_selection import StratifiedKFold

funcs.setup_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
dataset_base = 'datasets_DTI/datasets/'
dataset = 'DTI'
input_type = 'e'

lr = 1e-3
wd = 1e-5
b_size = 128

n_hidden = 128
num_epoches = 200


save_base = 'EDDTI-' + input_type
losses = nn.BCELoss()

skf = StratifiedKFold(n_splits=10, shuffle=True)

# get id map and features
dr_id_map, p_id_map, Drug_features, Protein_features = data_loader.Get_feature(dataset, input_type)
n_dr_feats, n_p_feats = len(Drug_features), len(Protein_features)
print('number of drug feature types: ', n_dr_feats)
print('number of protein feature types: ', n_p_feats)

# make path
model_save_path_base = 'model_10fold/' + save_base + '/' + dataset
funcs.Make_path(model_save_path_base)
# start
all_output_results = pd.DataFrame()
base_path = dataset_base + dataset

print('dataset: ', dataset)


P = np.loadtxt(base_path + '/DTI_P.csv', dtype=str, delimiter=",", skiprows=1)
N = np.loadtxt(base_path + '/DTI_N.csv', dtype=str, delimiter=",", skiprows=1)
X, Y = funcs.Get_sample(P, N, dr_id_map, p_id_map)


k = 0
for train_index, dev_index in skf.split(X, Y):
    fold_type = 'fold' + str(k + 1)
    print('lr: ', lr)
    print('wd: ', wd)
    print('batch_size: ', b_size)
    print('n_hidden: ', n_hidden)
    print('fold: ', fold_type)
    # data load path
    load_path = base_path + '/' + fold_type
    # model save path
    model_save_path = model_save_path_base + '/' + fold_type
    funcs.Make_path(model_save_path)

    train_X, dev_X = X[train_index], X[dev_index]
    train_Y, dev_Y = Y[train_index], Y[dev_index]

    # get loader
    train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
    dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)


    for m in range(n_dr_feats):
        for n in range(n_p_feats):
            n_dr_f = len(Drug_features[m][0])
            n_p_f = len(Protein_features[n][0])
            print('drug feature length: ', n_dr_f)
            print('protein feature length: ', n_p_f)
            model = DNNNet(n_dr_f, n_p_f, n_hidden).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            model_number = str(m * n_p_feats + n)
            print('model number: ', model_number)
            best_auc, best_epoch = 0, 0
            drug_feature = Drug_features[m]
            protein_feature = Protein_features[n]
            # train
            for epoch in range(num_epoches):
                train_loss = 0
                train_scores, train_scores_label, train_labels = [], [], []
                for step, (batch_x, batch_y) in enumerate(train_loader):
                    model.train()
                    b_x = batch_x.long().to(device)
                    b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                    b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                    b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                    output = model(drug_feature, protein_feature, b_x_dr, b_x_p)
                    score = torch.squeeze(output, dim=1)
                    loss = losses(score, b_y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()
                    scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                    train_scores = np.concatenate((train_scores, scores))
                    train_labels = np.concatenate((train_labels, label))

                train_avloss = train_loss / len(train_loader)
                train_auc = skm.roc_auc_score(train_labels, train_scores)
                # valid
                dev_scores, dev_labels = [], []
                with torch.no_grad():
                    for step, (batch_x, batch_y) in enumerate(dev_loader):
                        model.eval()
                        b_x = batch_x.long().to(device)
                        b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                        b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                        b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                        output = model(drug_feature, protein_feature, b_x_dr, b_x_p)
                        score = torch.squeeze(output, dim=1)
                        scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                        dev_scores = np.concatenate((dev_scores, scores))
                        dev_labels = np.concatenate((dev_labels, label))
                    dev_auc = skm.roc_auc_score(dev_labels, dev_scores)
                    dev_aupr = skm.average_precision_score(dev_labels, dev_scores)

                    print(
                        'epoch:{},Train Loss: {:.4f},Train Auc: {:.4f}, Dev Auc: {:.4f},Dev Aupr: {:.4f}'.format(
                            epoch, train_avloss, train_auc, dev_auc, dev_aupr))

                    if dev_auc >= best_auc:
                        best_model = model
                        best_auc = dev_auc
                        best_dev_labels = dev_labels
                        best_epoch = epoch
                        best_dev_scores = dev_scores
                        torch.save(model.state_dict(),
                                   model_save_path + '/model' + str(model_number) + '.pt')
            if model_number == '0':
                val_labels_pandas = pd.DataFrame(best_dev_labels)
                val_labels_pandas.to_csv(model_save_path + '/val_labels.csv', index=False)
            val_scores_pandas = pd.DataFrame(best_dev_scores)
            val_scores_pandas.to_csv(model_save_path + '/val_scores' + str(model_number) + '.csv',
                                     index=False)

            print('best_epoch', best_epoch)
            print('best_dev_AUC:', best_auc)
    k = k + 1

# test
test_drug = np.loadtxt(base_path + '/Drug_id.csv', dtype=str, delimiter=",", skiprows=1)
test_protein = np.loadtxt(base_path + '/Protein_id.csv', dtype=str, delimiter=",", skiprows=1)
test_X = []
test_Y = []
for i in range(len(test_drug)):
    for j in range(len(test_protein)):
        test_X.append([dr_id_map[test_drug[i]], p_id_map[test_protein[j]]])
        test_Y.append([0])
test_X, test_Y = np.array(test_X), np.array(test_Y)
test_loader = funcs.get_test_loader(test_X, test_Y, b_size=len(test_protein))


all_scores = []

for k in range(10):
    output_scores = []
    fold_type = 'fold' + str(k + 1)
    print(fold_type)
    model_save_path = model_save_path_base + '/' + fold_type
    for m in range(n_dr_feats):
        for n in range(n_p_feats):
            n_dr_f = len(Drug_features[m][0])
            n_p_f = len(Protein_features[n][0])
            model_number = str(m * n_p_feats + n)
            print(model_number)
            test_model = DNNNet(n_dr_f, n_p_f, n_hidden).to(device)
            test_model.load_state_dict(
                torch.load(model_save_path + '/model' + str(model_number) + '.pt'))

            drug_feature = Drug_features[m]
            protein_feature = Protein_features[n]

            model_test_scores = []
            with torch.no_grad():
                for step, (batch_x, batch_y) in enumerate(test_loader):
                    test_model.eval()
                    b_x = batch_x.long().to(device)
                    b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                    b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                    output = test_model(drug_feature, protein_feature, b_x_dr, b_x_p)
                    score = torch.squeeze(output, dim=1)
                    scores = score.cpu().detach().numpy()
                    model_test_scores.append(scores)
            if model_number == '0':
                output_scores = np.array(model_test_scores)
            else:
                output_scores = np.add(output_scores, np.array(model_test_scores))
    output_scores = output_scores/(n_dr_feats*n_p_feats)
    if k == 0:
        all_scores = output_scores
    else:
        all_scores = np.add(all_scores, output_scores)

all_scores = all_scores/10
all_output_pandas = pd.DataFrame(all_scores)
all_output_pandas.to_csv("case study/All_scores_10fold_"+input_type+".csv", index=False)


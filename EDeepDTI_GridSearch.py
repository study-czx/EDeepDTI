import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import funcs
import pandas as pd
import data_loader
import pytorch_lightning as pl

hidden1, hidden2, hidden3 = 256, 128, 64

class DNNNet(pl.LightningModule):
    def __init__(self, n_dr_f, n_protein_f, n_hidden):
        super(DNNNet, self).__init__()
        self.drug_hidden_layer = nn.Sequential(nn.Linear(in_features=n_dr_f, out_features=n_hidden), nn.ReLU())
        self.protein_hidden_layer = nn.Sequential(nn.Linear(in_features=n_protein_f, out_features=n_hidden), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(n_hidden * 2, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden2, hidden3), nn.BatchNorm1d(hidden3), nn.ReLU())
        self.output = nn.Linear(hidden3, 1)
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, Drug_feature, Protein_feature, x_dr, x_p):
        dr_feat, p_feat = Drug_feature[x_dr].squeeze(1), Protein_feature[x_p].squeeze(1)
        h_dr = self.drug_hidden_layer(dr_feat)
        h_p = self.protein_hidden_layer(p_feat)
        h_dr_d = torch.cat((h_dr, h_p), dim=1)
        h_hidden = self.fc3(self.fc2(self.fc1(h_dr_d)))
        out = self.sigmoid(self.output(h_hidden))
        return out

funcs.setup_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
dataset_base = 'datasets_DTI/datasets/'
dataset = 'DTI'
predict_type = '5_fold'
input_type = 'e'

# Grid search
# 针对单个值进行，其余值固定
# lrs = [1e-3, 1e-4, 1e-5]
# wds = [1e-3, 1e-4, 1e-5, 1e-6, 0]
lrs = [1e-3]
wds = [1e-5]
num_epoches = 200

b_sizes = [128]
n_hiddens = [128]


save_base = 'EDDTI-' + input_type
losses = nn.BCELoss()

# make path
model_save_path_base = 'models_grid/' + save_base + '/' + dataset + '/' + predict_type
funcs.Make_path(model_save_path_base)

# get id map and features
dr_id_map, p_id_map, Drug_features, Protein_features = data_loader.Get_feature(dataset, input_type)
n_dr_feats, n_p_feats = len(Drug_features), len(Protein_features)

print('number of drug feature types: ', n_dr_feats)
print('number of protein feature types: ', n_p_feats)
print('number of base learners: ', n_dr_feats * n_p_feats)

drug_name_list = list(Drug_features.keys())
protein_name_list = list(Protein_features.keys())
drug_embedding_list = list(Drug_features.values())
protein_embedding_list = list(Protein_features.values())
print(drug_name_list)
print(protein_name_list)
del Drug_features, Protein_features



# start
all_output_results = pd.DataFrame()
for lr in lrs:
    for wd in wds:
        for b_size in b_sizes:
            for n_hidden in n_hiddens:
                base_path = dataset_base + dataset + '/' + predict_type
                for k in range(5):
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

                    train_P = np.loadtxt(load_path + '/train_P.csv', dtype=str, delimiter=',', skiprows=1)
                    dev_P = np.loadtxt(load_path + '/dev_P.csv', dtype=str, delimiter=',', skiprows=1)
                    test_P = np.loadtxt(load_path + '/test_P.csv', dtype=str, delimiter=',', skiprows=1)
                    train_N = np.loadtxt(load_path + '/train_N.csv', dtype=str, delimiter=',', skiprows=1)
                    dev_N = np.loadtxt(load_path + '/dev_N.csv', dtype=str, delimiter=',', skiprows=1)
                    test_N = np.loadtxt(load_path + '/test_N.csv', dtype=str, delimiter=',', skiprows=1)
                    print('number of DTI: ', len(train_P), len(dev_P), len(test_P))
                    print('number of Negative DTI ', len(train_N), len(dev_N), len(test_N))
                    # trans samples to id map and get X Y
                    train_X, train_Y = funcs.Get_sample(train_P, train_N, dr_id_map, p_id_map)
                    dev_X, dev_Y = funcs.Get_sample(dev_P, dev_N, dr_id_map, p_id_map)
                    test_X, test_Y = funcs.Get_sample(test_P, test_N, dr_id_map, p_id_map)
                    # get loader
                    train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
                    dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)
                    test_loader = funcs.get_test_loader(test_X, test_Y, len(test_P) + len(test_N))

                    for m in range(n_dr_feats):
                        for n in range(n_p_feats):
                            n_dr_f = len(drug_embedding_list[m][0])
                            n_p_f = len(protein_embedding_list[n][0])
                            print('drug feature is {}, length is {}'.format(drug_name_list[m], n_dr_f))
                            print('protein feature is {}, length is {}'.format(protein_name_list[n], n_p_f))
                            drug_feature = torch.as_tensor(torch.from_numpy(drug_embedding_list[m]),dtype=torch.float32).to(device)
                            protein_feature = torch.as_tensor(torch.from_numpy(protein_embedding_list[n]),dtype=torch.float32).to(device)
                            model = DNNNet(n_dr_f, n_p_f, n_hidden).to(device)
                            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                            model_number = str(m * n_p_feats + n)
                            print('model number: ', model_number)
                            best_auc, best_epoch = 0, 0
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
                            # test
                            test_scores, test_scores_label, test_labels = [], [], []
                            test_model = DNNNet(n_dr_f, n_p_f, n_hidden).to(device)
                            test_model.load_state_dict(
                                torch.load(model_save_path + '/model' + str(model_number) + '.pt'))
                            with torch.no_grad():
                                for step, (batch_x, batch_y) in enumerate(test_loader):
                                    test_model.eval()
                                    b_x = batch_x.long().to(device)
                                    b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                                    b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                                    b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                                    output = test_model(drug_feature, protein_feature, b_x_dr, b_x_p)
                                    score = torch.squeeze(output, dim=1)
                                    scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                                    test_scores = np.concatenate((test_scores, scores))
                                    test_labels = np.concatenate((test_labels, label))

                            if model_number == '0':
                                test_labels_pandas = pd.DataFrame(test_labels)
                                test_labels_pandas.to_csv(model_save_path + '/test_labels.csv', index=False)
                            test_scores_pandas = pd.DataFrame(test_scores)
                            test_scores_pandas.to_csv(model_save_path + '/test_scores' + str(model_number) + '.csv',
                                                      index=False)

                output_score = np.zeros(shape=(7, 5))
                for k in range(5):
                    fold_type = 'fold' + str(k + 1)
                    model_save_path = model_save_path_base + '/' + fold_type
                    all_labels = np.loadtxt(model_save_path + '/test_labels.csv', skiprows=1)
                    all_output_scores = []
                    for i in range(n_dr_feats):
                        for j in range(n_p_feats):
                            model_number = i * n_p_feats + j
                            this_scores = np.loadtxt(model_save_path + '/test_scores' + str(model_number) + '.csv',
                                                     skiprows=1)
                            all_output_scores.append(this_scores)
                    all_output_scores = list(np.mean(np.array(all_output_scores), axis=0))
                    best_test = funcs.get_metric(all_labels, all_output_scores)
                    for m in range(7):
                        output_score[m][k] = best_test[m]
                mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision = funcs.show_metric(
                    output_score, base_path, input_type)

                this_dict = {'lr': lr, 'wd': wd, 'b_size': b_size, 'n_hidden': n_hidden, 'mean_acc': mean_acc,
                             'mean_auc': mean_auc, 'mean_aupr': mean_aupr, 'mean_mcc': mean_mcc,
                             'mean_f1': mean_f1, 'mean_recall': mean_recall, 'mean_precision': mean_precision,
                             'hidden1': hidden1, 'hidden2': hidden2, 'hidden3': hidden3}
                record = pd.DataFrame.from_dict(this_dict, orient='index').T
                print(record)
                if all_output_results.empty:
                    all_output_results = record
                else:
                    all_output_results = pd.concat([all_output_results, record])
                all_output_results.to_csv('EDDTI_e_all_records.csv', index=False)

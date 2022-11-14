import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import torch.nn.functional as F
import funcs
import pandas as pd
import data_loader
from sklearn.model_selection import StratifiedKFold

funcs.setup_seed(1)

b_size, n_hidden = 128, 128
lr, wd = 1e-3, 1e-4
num_epoches = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Drug_id, Protein_id = data_loader.Get_id()
n_drugs, n_proteins = len(Drug_id), len(Protein_id)
dr_id_map, p_id_map = funcs.id_map(Drug_id), funcs.id_map(Protein_id)
print("number of Drug: ", n_drugs)
print("number of Protein ", n_proteins)
n_drugs, n_proteins = len(Drug_id), len(Protein_id)

# Dr_finger = data_loader.Get_finger()
# P_seq = data_loader.Get_seq()

Dr_sim = data_loader.Get_drug_sim()
P_sim = data_loader.Get_protein_sim()

def Trans_feature(Feature):
    for i in Feature:
        Feature[i] = torch.as_tensor(torch.from_numpy(Feature[i]), dtype=torch.float32).to(device)
    return Feature

Dr_sim = Trans_feature(Dr_sim)
P_sim = Trans_feature(P_sim)
Drug_features = [Dr_sim['ecfp4'], Dr_sim['fcfp4'], Dr_sim['pubchem'], Dr_sim['maccs'], Dr_sim['rdk'], Dr_sim['DDI'], Dr_sim['Dr_D']]
Protein_features = [P_sim['seq'], P_sim['MF'], P_sim['BP'], P_sim['CC'], P_sim['PPI'], P_sim['PPI2'], P_sim['P_D']]
n_dr_feats, n_p_feats = 7, 7

class DNNNet(nn.Module):
    def __init__(self, n_dr_f, n_protein_f):
        super(DNNNet, self).__init__()
        self.drug_hidden_layer1 = nn.Sequential(nn.Linear(in_features=n_dr_f, out_features=n_hidden), nn.ReLU())
        self.protein_hidden_layer1 = nn.Sequential(nn.Linear(in_features=n_protein_f, out_features=n_hidden), nn.ReLU())
        self.connected_layer1 = nn.Sequential(nn.Linear(in_features=n_hidden * 2, out_features=256), nn.BatchNorm1d(num_features=256), nn.ReLU())
        self.connected_layer2 = nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.BatchNorm1d(num_features=128), nn.ReLU())
        self.connected_layer3 = nn.Sequential(nn.Linear(in_features=128, out_features=64), nn.BatchNorm1d(num_features=64), nn.ReLU())
        self.output = nn.Linear(in_features=64, out_features=1)
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, Drug_feature, Protein_feature, x_dr, x_p):
        dr_feat, p_feat = Drug_feature[x_dr].squeeze(1), Protein_feature[x_p].squeeze(1)
        h_dr = self.drug_hidden_layer1(dr_feat)
        h_p = self.protein_hidden_layer1(p_feat)
        h_dr_d = torch.cat((h_dr, h_p), dim=1)
        out = self.connected_layer1(h_dr_d)
        out = self.connected_layer2(out)
        out = self.connected_layer3(out)
        out = self.sigmoid(self.output(out))
        return out

P = np.loadtxt("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/DTI-benchmark_set/DTI_8020.csv", dtype=str, delimiter=",", skiprows=1)
N = np.loadtxt("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/negative samples/neg_DTI-net_8020.csv", dtype=str, delimiter=",", skiprows=1)
X, Y = funcs.Get_sample(P, N, dr_id_map, p_id_map)

skf = StratifiedKFold(n_splits=10, shuffle=True)

test_drug = np.loadtxt("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/DTI-benchmark_set/Drug_1409.csv", dtype=str, delimiter=",", skiprows=1)
test_protein = np.loadtxt("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/DTI-benchmark_set/Protein_1648.csv", dtype=str, delimiter=",", skiprows=1)
test_X = []
test_Y = []
for i in range(len(test_drug)):
    for j in range(len(test_protein)):
        test_X.append([dr_id_map[test_drug[i]], p_id_map[test_protein[j]]])
        test_Y.append([0])
test_X, test_Y = np.array(test_X), np.array(test_Y)

print("number of DTI: ", len(P))
print("number of Negative DTI ", len(N))

all_output_scores1, all_output_scores2, all_output_scores3, all_output_scores4, \
all_output_scores5, all_output_scores6, all_output_scores7, all_output_scores8, \
all_output_scores9, all_output_scores10 = [],[],[],[],[],[],[],[],[],[]

this_fold = 0
for train_index, dev_index in skf.split(X, Y):
    this_fold = this_fold + 1
    print("Fold: ", this_fold)
    X_train, X_dev = X[train_index], X[dev_index]
    Y_train, Y_dev = Y[train_index], Y[dev_index]
    train_loader = funcs.get_train_loader(X_train, Y_train, b_size)
    dev_loader = funcs.get_train_loader(X_dev, Y_dev, b_size)
    test_loader = funcs.get_test_loader(test_X, test_Y, b_size=len(test_protein))
    losses = nn.BCELoss()
    models = []
    for i in range(n_dr_feats):
        for j in range(n_p_feats):
            n_dr_f = len(Drug_features[i][0])
            n_p_f = len(Protein_features[j][0])
            models.append(DNNNet(n_dr_f, n_p_f).to(device))
    opts, schedulers = [], []
    # 初始化所有的优化器
    for model in models:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=40, verbose=False)
        opts.append(opt)
        schedulers.append(scheduler)
    best_auc, best_epoch, best_extra = 0, 0, 0
    best_test = []
    for epoch in range(num_epoches):
        train_losses = []
        for i in range(len(models)):
            train_losses.append(0)
        for step, (batch_x, batch_y) in enumerate(train_loader):
            b_x = batch_x.long().to(device)
            b_y = torch.squeeze(batch_y.float().to(device), dim=1)
            b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
            b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
            # 训练所有模型
            for i, model, opt in zip(range(len(models)), models, opts):
                model.train()
                drug_feature = Drug_features[int(i / n_p_feats)]
                protein_feature = Protein_features[i % n_p_feats]
                output = model(drug_feature, protein_feature, b_x_dr, b_x_p)
                score = torch.squeeze(output, dim=1)
                loss = losses(score, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_losses[i] += loss.item()
        all_train_loss = np.array(train_losses) / len(train_loader)
        final_train_losses = np.mean(all_train_loss, axis=0)
        with torch.no_grad():
            all_dev_aucs = []
            all_dev_scores = []
            for i in range(len(models)):
                all_dev_aucs.append(0)
                all_dev_scores.append([])
            dev_labels = []
            for step, (batch_x, batch_y) in enumerate(dev_loader):
                b_x = batch_x.long().to(device)
                b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                for i, model in zip(range(len(models)), models):
                    model.eval()
                    drug_feature = Drug_features[int(i / n_p_feats)]
                    protein_feature = Protein_features[i % n_p_feats]
                    output = model(drug_feature, protein_feature, b_x_dr, b_x_p)
                    score = torch.squeeze(output, dim=1)
                    scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                    all_dev_scores[i] = np.concatenate((all_dev_scores[i], scores))
                dev_labels = np.concatenate((dev_labels, b_y.cpu().detach().numpy()))
            for i in range(len(models)):
                scheduler = schedulers[i]
                all_dev_aucs[i] = skm.roc_auc_score(dev_labels, all_dev_scores[i])
                scheduler.step(all_dev_aucs[i])
            # print(all_dev_aucs)
            mean_all_dev_scores = np.mean(all_dev_scores, axis=0)
            final_dev_auc = skm.roc_auc_score(dev_labels, mean_all_dev_scores)
            if final_dev_auc >= best_auc:
                best_auc = final_dev_auc
                best_epoch = epoch
                for i in range(len(models)):
                    torch.save(models[i], "test_model/fold" + str(this_fold) + "/model" + str(i+1) + ".pt")
        print('epoch:{},Train Loss: {:.4f}, Dev Auc: {:.4f}'.format(epoch, final_train_losses, final_dev_auc))

    my_models = []
    for i in range(len(models)):
        this_model = torch.load("test_model/fold" + str(this_fold) + "/model" + str(i+1) + ".pt")
        my_models.append(this_model)
    with torch.no_grad():
        all_test_scores = []
        for i in range(len(models)):
            all_test_scores.append([])
        test_labels = []
        for step, (batch_x, batch_y) in enumerate(test_loader):
            b_x = batch_x.long().to(device)
            b_y = torch.squeeze(batch_y.float().to(device), dim=1)
            b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
            b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
            for i, model in zip(range(len(my_models)), my_models):
                model.eval()
                drug_feature = Drug_features[int(i / n_p_feats)]
                protein_feature = Protein_features[i % n_p_feats]
                output = model(drug_feature, protein_feature, b_x_dr, b_x_p)
                score = torch.squeeze(output, dim=1)
                scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                all_test_scores[i] = np.concatenate((all_test_scores[i], scores))
        mean_all_test_scores = np.mean(all_test_scores, axis=0)
        globals()['all_output_scores' + str(this_fold)].append(scores)
        globals()['all_output_scores' + str(this_fold)] = np.array(globals()['all_output_scores' + str(this_fold)])

all_output_scores = all_output_scores1 + all_output_scores2 + all_output_scores3 + all_output_scores4 + all_output_scores5 + all_output_scores6 + all_output_scores7 + all_output_scores8 + all_output_scores9 + all_output_scores10
all_output_scores = all_output_scores / 10
all_output_pandas = pd.DataFrame(all_output_scores)
all_output_pandas.to_csv("All_scores_10fold.csv", index=False)

import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import torch.nn.functional as F
import funcs
import pandas as pd
import data_loader_CPI

funcs.setup_seed(1)

# 选择负样本的方式，N_1，N_2，N_3，N_5，N_7，N_9, N_10
# n_length = "N_0"
types = ["random","new_drug","new_protein", "new_drug_protein"]
# datasets = ["CPI_dataset/"]
datasets = ["CPI dataset/"]

b_size, n_hidden = 128, 128
lr, wd = 1e-3, 1e-4
num_epoches = 200
# 使用GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Drug_id, Protein_id = data_loader_CPI.Get_CPI_id()
n_drugs, n_proteins = len(Drug_id), len(Protein_id)
dr_id_map, p_id_map = funcs.id_map(Drug_id), funcs.id_map(Protein_id)
print("number of Drug: ", n_drugs)
print("number of Protein ", n_proteins)
n_drugs, n_proteins = len(Drug_id), len(Protein_id)

Dr_finger = data_loader_CPI.Get_CPI_finger()
P_seq = data_loader_CPI.Get_CPI_seq()

# Dr_sim = data_loader_CPI.Get_drug_sim()
# P_sim = data_loader_CPI.Get_protein_sim()

def Trans_feature(Feature):
    for i in Feature:
        Feature[i] = torch.as_tensor(torch.from_numpy(Feature[i]), dtype=torch.float32).to(device)
    return Feature

Dr_finger = Trans_feature(Dr_finger)
P_seq = Trans_feature(P_seq)
Drug_features = [Dr_finger['ecfp4'], Dr_finger['fcfp4'], Dr_finger['pubchem'], Dr_finger['maccs'], Dr_finger['rdk']]
Protein_features = [P_seq['KSCTriad'], P_seq['CKSAAP'], P_seq['TPC'], P_seq['PAAC'], P_seq['CTD']]
n_dr_feats, n_p_feats = 5, 5

class DNNNet(nn.Module):
    def __init__(self, n_dr_f, n_protein_f):
        super(DNNNet, self).__init__()
        self.drug_hidden_layer1 = nn.Sequential(nn.Linear(in_features=n_dr_f, out_features=n_hidden), nn.ReLU())
        self.protein_hidden_layer1 = nn.Sequential(nn.Linear(in_features=n_protein_f, out_features=n_hidden), nn.ReLU())
        self.connected_layer1 = nn.Sequential(nn.Linear(in_features=n_hidden * 2, out_features=256),
                                              nn.BatchNorm1d(num_features=256), nn.ReLU())
        self.connected_layer2 = nn.Sequential(nn.Linear(in_features=256, out_features=128),
                                              nn.BatchNorm1d(num_features=128), nn.ReLU())
        self.connected_layer3 = nn.Sequential(nn.Linear(in_features=128, out_features=64),
                                              nn.BatchNorm1d(num_features=64), nn.ReLU())
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


for dataset in datasets:
    for type in types:
        output_score = np.zeros(shape=(3, 5))
        for k in range(5):
            seed_type = "seed" + str(k + 1)
            train_P = np.loadtxt(dataset + seed_type + "/" + type + "/train_P.csv", dtype=str, delimiter=",",
                                 skiprows=1)
            dev_P = np.loadtxt(dataset + seed_type + "/" + type + "/dev_P.csv", dtype=str, delimiter=",", skiprows=1)
            test_P = np.loadtxt(dataset + seed_type + "/" + type + "/test_P.csv", dtype=str, delimiter=",", skiprows=1)
            train_N = np.loadtxt(dataset + seed_type + "/" + type + "/train_N.csv", dtype=str, delimiter=",",
                                 skiprows=1)
            dev_N = np.loadtxt(dataset + seed_type + "/" + type + "/dev_N.csv", dtype=str, delimiter=",", skiprows=1)
            test_N = np.loadtxt(dataset + seed_type + "/" + type + "/test_N.csv", dtype=str, delimiter=",", skiprows=1)
            print("number of DTI: ", len(train_P), len(dev_P), len(test_P))
            print("number of Negative DTI ", len(train_N), len(dev_N), len(test_N))
            train_X, train_Y = funcs.Get_sample(train_P, train_N, dr_id_map, p_id_map)
            dev_X, dev_Y = funcs.Get_sample(dev_P, dev_N, dr_id_map, p_id_map)
            test_X, test_Y = funcs.Get_sample(test_P, test_N, dr_id_map, p_id_map)
            train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
            dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)
            test_loader = funcs.get_test_loader(test_X, test_Y, b_size)
            losses = nn.BCELoss()
            models = []
            for i in range(n_dr_feats):
                for j in range(n_p_feats):
                    n_dr_f = len(Drug_features[i][0])
                    n_p_f = len(Protein_features[j][0])
                    models.append(DNNNet(n_dr_f, n_p_f).to(device))
            best_auc, best_epoch = 0, 0
            best_test = []
            opts, schedulers = [], []
            # 初始化所有的优化器
            for model in models:
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=40, verbose=False)
                opts.append(opt)
                schedulers.append(scheduler)
            # 开始迭代
            for epoch in range(num_epoches):
                # 开始训练
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

                # 验证
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
                    # 测试
                    all_test_scores = []
                    for i in range(len(models)):
                        all_test_scores.append([])
                    test_labels = []
                    for step, (batch_x, batch_y) in enumerate(test_loader):
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
                            all_test_scores[i] = np.concatenate((all_test_scores[i], scores))
                        test_labels = np.concatenate((test_labels, b_y.cpu().detach().numpy()))
                    mean_all_test_scores = np.mean(all_test_scores, axis=0)
                    test_scores_label = funcs.computer_label(mean_all_test_scores, 0.5)
                    final_test_acc = skm.accuracy_score(test_labels, test_scores_label)
                    final_test_auc = skm.roc_auc_score(test_labels, mean_all_test_scores)
                    final_test_aupr = skm.average_precision_score(test_labels, mean_all_test_scores)

                    # 额外测试集
                print(
                    'epoch:{},Train Loss: {:.4f},Dev Auc: {:.4f}, Test Acc: {:.4f},Test Auc: {:.4f},TestAUPR: {:.4f}'
                        .format(epoch, final_train_losses, final_dev_auc, final_test_acc, final_test_auc,
                                final_test_aupr))
                if final_dev_auc >= best_auc:
                    best_auc = final_dev_auc
                    best_epoch = epoch
                    best_test = [round(final_test_acc, 4), round(final_test_auc, 4), round(final_test_aupr, 4)]

            print("best_dev_AUC:", best_auc)
            print("best_epoch", best_epoch)
            print("test_out", best_test)
            output_score[0][k], output_score[1][k], output_score[2][k] = best_test[0], best_test[1], best_test[2]

        print(output_score)
        mean_acc, mean_auc, mean_mcc = np.nanmean(output_score[0]), np.nanmean(output_score[1]), np.nanmean(
            output_score[2])
        std_acc, std_auc, std_mcc = np.nanstd(output_score[0]), np.nanstd(output_score[1]), np.nanstd(output_score[2])
        print(mean_acc, mean_auc, mean_mcc)
        print(std_acc, std_auc, std_mcc)

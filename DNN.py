import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import funcs
import pandas as pd
import data_loader

funcs.setup_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_base = 'datasets_DTI/datasets/'

dataset = 'DTI'
b_size, n_hidden = 128, 128
lr, wd = 1e-4, 1e-4
num_epoches = 200

predict_types = ['5_fold']
losses = nn.BCELoss()


class DNNNet(nn.Module):
    def __init__(self, n_dr_f, n_protein_f):
        super(DNNNet, self).__init__()
        self.drug_hidden_layer = nn.Sequential(nn.Linear(in_features=n_dr_f, out_features=n_hidden), nn.ReLU())
        self.protein_hidden_layer = nn.Sequential(nn.Linear(in_features=n_protein_f, out_features=n_hidden), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(n_hidden * 2, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.output = nn.Linear(64, 1)
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


def Running():
    # get id map and features
    Drug_id, Protein_id = data_loader.Get_id(dataset)
    n_drugs, n_proteins = len(Drug_id), len(Protein_id)
    dr_id_map, p_id_map = funcs.id_map(Drug_id), funcs.id_map(Protein_id)
    drug_feature = np.loadtxt('datasets/datasets/DTI/drug_embedding/grover_emb_both.csv', dtype=float, delimiter=',')
    protein_feature = np.loadtxt('datasets/datasets/DTI/protein_embedding/ESM2_emb.csv', dtype=float, delimiter=',')
    drug_feature = torch.as_tensor(torch.from_numpy(drug_feature), dtype=torch.float32).to(device)
    protein_feature = torch.as_tensor(torch.from_numpy(protein_feature), dtype=torch.float32).to(device)

    # start
    output_score = np.zeros(shape=(8, 5))
    for predict_type in predict_types:
        for k in range(5):
            fold_type = 'fold' + str(k + 1)
            load_path = dataset_base + dataset + '/' + predict_type + '/' + fold_type
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
            best_dev_scores, best_dev_labels = [], []
            # init all models, opts, schedulers

            n_dr_f = len(drug_feature[0])
            n_p_f = len(protein_feature[0])
            print('drug feature length: ', n_dr_f)
            print('protein feature length: ', n_p_f)
            model = DNNNet(n_dr_f, n_p_f).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=40,
            #                                                        verbose=False)

            best_auc, best_epoch = 0, 0
            best_test = []
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
                train_scores_label = funcs.computer_label(train_scores, 0.5)
                train_avloss = train_loss / len(train_loader)
                train_acc = skm.accuracy_score(train_labels, train_scores_label)
                train_auc = skm.roc_auc_score(train_labels, train_scores)

                dev_scores, dev_labels = [], []
                test_scores, test_scores_label, test_labels = [], [], []
                with torch.no_grad():
                    # validation
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
                    # for P:N != 1:1, use dev_aupr
                    # dev_aupr = skm.average_precision_score(dev_labels, dev_scores)
                    # scheduler.step(dev_auc)
                    # testing
                    for step, (batch_x, batch_y) in enumerate(test_loader):
                        model.eval()
                        b_x = batch_x.long().to(device)
                        b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                        b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                        b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                        output = model(drug_feature, protein_feature, b_x_dr, b_x_p)
                        score = torch.squeeze(output, dim=1)
                        scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                        test_scores = np.concatenate((test_scores, scores))
                        test_labels = np.concatenate((test_labels, label))
                    test_scores_label = funcs.computer_label(test_scores, 0.5)
                    test_acc = skm.accuracy_score(test_labels, test_scores_label)
                    test_auc = skm.roc_auc_score(test_labels, test_scores)
                    test_aupr = skm.average_precision_score(test_labels, test_scores)
                    test_mcc = skm.matthews_corrcoef(test_labels, test_scores_label)
                    test_F1 = skm.f1_score(test_labels, test_scores_label)
                    test_recall = skm.recall_score(test_labels, test_scores_label)
                    test_precision = skm.precision_score(test_labels, test_scores_label)
                print(
                    'epoch:{},Train Loss: {:.4f},Train Acc: {:.4f},Train Auc: {:.4f},Dev Aupr: {:.4f}, Test Acc: {:.4f},Test Auc: {:.4f},TestAUPR: {:.4f}'
                    .format(epoch, train_avloss, train_acc, train_auc, dev_auc, test_acc, test_auc, test_aupr))
                if dev_auc >= best_auc:
                    best_auc = dev_auc
                    best_epoch = epoch
                    best_test = [format(test_acc, '.4f'), format(test_auc, '.4f'), format(test_aupr, '.4f'),
                                 format(test_mcc, '.4f'),
                                 format(test_F1, '.4f'), format(test_recall, '.4f'), format(test_precision, '.4f')]
            print("best_dev_AUC:", best_auc)
            print("best_epoch", best_epoch)
            print("test_out", best_test)
            output_score[0][k], output_score[1][k], output_score[2][k], output_score[3][k], output_score[4][k], \
            output_score[5][k], output_score[6][k], output_score[7][k] = best_auc, best_test[0], \
                                                                         best_test[1], best_test[2], best_test[3], \
                                                                         best_test[4], best_test[5], best_test[6]

        print(output_score)
        mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision = \
            np.nanmean(output_score[1]), np.nanmean(output_score[2]), np.nanmean(output_score[3]), np.nanmean(
                output_score[4]), \
            np.nanmean(output_score[5]), np.nanmean(output_score[6]), np.nanmean(output_score[7])
        std_acc, std_auc, std_aupr, std_mcc, std_f1, std_recall, std_precision = \
            np.nanstd(output_score[1]), np.nanstd(output_score[2]), np.nanstd(output_score[3]), np.nanstd(
                output_score[4]), \
            np.nanstd(output_score[5]), np.nanstd(output_score[6]), np.nanstd(output_score[7])
        print(mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision)
        print(std_acc, std_auc, std_aupr, std_mcc, std_f1, std_recall, std_precision)

Running()
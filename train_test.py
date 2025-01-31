import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as skm
import pandas as pd
from model import DNNNet
import funcs
import os
import time

funcs.setup_seed(1)

def train_model(drug_feature, protein_feature, model, opt, losses, train_loader, dev_loader, num_epoches, device, model_save_path, model_number, dataset):
    best_auc, best_aupr, best_epoch = 0, 0, 0
    train_time_all, dev_time_all = 0, 0
    for epoch in range(num_epoches):
        time_start = time.time()
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

        # train_avloss = train_loss / len(train_loader)
        # train_auc = skm.roc_auc_score(train_labels, train_scores)
        time_second = time.time()
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
            if dataset == 'DTI':
                dev_auc = skm.roc_auc_score(dev_labels, dev_scores)
                if dev_auc >= best_auc:
                    best_model = model
                    best_auc = dev_auc
                    best_dev_labels = dev_labels
                    best_epoch = epoch
                    best_dev_scores = dev_scores
            else:
                dev_aupr = skm.average_precision_score(dev_labels, dev_scores)
                if dev_aupr >= best_aupr:
                    best_model = model
                    best_aupr = dev_aupr
                    best_dev_labels = dev_labels
                    best_epoch = epoch
                    best_dev_scores = dev_scores

        time_end = time.time()
        train_time_all = train_time_all + (time_second-time_start)
        dev_time_all = dev_time_all + (time_end - time_second)
    torch.save(best_model.state_dict(), model_save_path + '/model' + str(model_number) + '.pt')

    if model_number == '0':
        val_labels_pandas = pd.DataFrame(best_dev_labels)
        val_labels_pandas.to_csv(model_save_path + '/val_labels.csv', index=False)
    val_scores_pandas = pd.DataFrame(best_dev_scores)
    val_scores_pandas.to_csv(model_save_path + '/val_scores' + str(model_number) + '.csv', index=False)
    if dataset == 'DTI':
        print('model {}, best_epoch is {}, best validation AUC is {}'.format(model_number, best_epoch, best_auc))
    else:
        print('model {}, best_epoch is {}, best validation AUPR is {}'.format(model_number, best_epoch, best_aupr))
    return train_time_all, dev_time_all

def test_model(n_dr_f, n_p_f, model_save_path, model_number, test_loader, drug_feature, protein_feature, n_hidden, device, data_save_path):
    test_scores, test_scores_label, test_labels = [], [], []
    test_model = DNNNet(n_dr_f, n_p_f, n_hidden).to(device)
    test_model.load_state_dict(torch.load(model_save_path + '/model' + str(model_number) + '.pt'))
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
        test_labels_pandas.to_csv(data_save_path + '/test_labels.csv', index=False)
    test_scores_pandas = pd.DataFrame(test_scores)
    test_scores_pandas.to_csv(data_save_path + '/test_scores' + str(model_number) + '.csv',
                              index=False)

def get_result(model_save_path_base, n_dr_feats, n_p_feats):
    output_score = np.zeros(shape=(7, 5))
    for k in range(5):
        fold_type = 'fold' + str(k + 1)
        model_save_path = model_save_path_base + '/' + fold_type
        all_labels = np.loadtxt(model_save_path + '/test_labels.csv', skiprows=1)
        all_output_scores = []
        for i in range(n_dr_feats):
            for j in range(n_p_feats):
                model_number = i * n_p_feats + j
                this_scores = np.loadtxt(model_save_path + '/test_scores' + str(model_number) + '.csv',skiprows=1)
                all_output_scores.append(this_scores)
        all_output_scores = list(np.mean(np.array(all_output_scores), axis=0))
        best_test = funcs.get_metric(all_labels, all_output_scores)
        for m in range(7):
            output_score[m][k] = best_test[m]
    output_score2 = pd.DataFrame(output_score).T
    output_score2.columns = ['ACC', 'AUC', 'AUPR', 'MCC', 'F1', 'Recall', 'Precision']
    pd_out = output_score2[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
    return pd_out

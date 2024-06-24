import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import funcs
import pandas as pd
import data_loader
from model import DNNNet

funcs.setup_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
dataset_base = 'datasets_for_PPI/'
datasets = ['Human']
# datasets = ['S.cerevisiae']

fold_number = 5

lr = 1e-3
wd = 1e-5
b_size = 128

n_hidden = 128
num_epoches = 200

losses = nn.BCELoss()

def show_metric(output_score, result_path):
    print(output_score)
    mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision = np.nanmean(
        output_score[0]), np.nanmean(output_score[1]), np.nanmean(output_score[2]), np.nanmean(
        output_score[3]), np.nanmean(output_score[4]), np.nanmean(output_score[5]), np.nanmean(
        output_score[6])
    std_acc, std_auc, std_aupr, std_mcc, std_f1, std_recall, std_precision = np.nanstd(
        output_score[0]), np.nanstd(
        output_score[1]), np.nanstd(output_score[2]), np.nanstd(output_score[3]), np.nanstd(
        output_score[4]), np.nanstd(output_score[5]), np.nanstd(output_score[6])
    print(mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision)
    print(std_acc, std_auc, std_aupr, std_mcc, std_f1, std_recall, std_precision)
    pd_output = pd.DataFrame(output_score)
    pd_output.to_csv(result_path + '_score.csv', index=False)
    return mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision


def Get_protein_embedding(data_type):
    emb_feature_path_p = dataset_base + 'feature/' + data_type + '/'
    esm2 = np.loadtxt(emb_feature_path_p + 'ESM2_emb.csv', dtype=float, delimiter=',')
    protein_bert = np.loadtxt(emb_feature_path_p + 'Protein_bert_emb.csv', dtype=float, delimiter=',')
    prottrans = np.loadtxt(emb_feature_path_p + 'ProtTrans_emb.csv', dtype=float, delimiter=',')
    tape = np.loadtxt(emb_feature_path_p + 'TAPE_emb.csv', dtype=float, delimiter=',')
    esm2_max = np.loadtxt(emb_feature_path_p + 'ESM2_emb_max.csv', dtype=float, delimiter=',')
    protein_bert_max = np.loadtxt(emb_feature_path_p + 'Protein_bert_emb_max.csv', dtype=float, delimiter=',')
    prottrans_max = np.loadtxt(emb_feature_path_p + 'ProtTrans_emb_max.csv', dtype=float, delimiter=',')
    tape_max = np.loadtxt(emb_feature_path_p + 'TAPE_emb_max.csv', dtype=float, delimiter=',')
    P_embedding = {'esm2': esm2, 'protein_bert': protein_bert, 'prottrans': prottrans, 'tape': tape,
                   'esm2_max': esm2_max, 'protein_bert_max': protein_bert_max, 'prottrans_max': prottrans_max,
                   'tape_max': tape_max}
    P_embedding = data_loader.Trans_feature(P_embedding)
    Protein_features = [P_embedding['esm2'], P_embedding['protein_bert'], P_embedding['prottrans'],
                        P_embedding['tape'], P_embedding['esm2_max'], P_embedding['protein_bert_max'],
                        P_embedding['prottrans_max'], P_embedding['tape_max']]
    return Protein_features


def Get_sample(DTI, N_DTI):
    P_list, N_list = [], []
    P_label, N_label = [], []
    for i in range(len(DTI)):
        P_list.append([DTI[i][0], DTI[i][1]])
        P_label.append([1])
    for j in range(len(N_DTI)):
        N_list.append([N_DTI[j][0], N_DTI[j][1]])
        N_label.append([0])
    X = np.concatenate((P_list, N_list))
    Y = np.concatenate((P_label, N_label))
    return X, Y


for dataset in datasets:
    if dataset == 'H.pylori':
        b_size = 32
    else:
        b_size = 128
    protein_features = Get_protein_embedding(dataset)
    n_p_feats = len(protein_features)
    print('number of protein feature types: ', n_p_feats)

    model_save_path_base = 'models_PPI/' + dataset
    funcs.Make_path(model_save_path_base)

    all_output_results = pd.DataFrame()
    for k in range(fold_number):
        fold_type = 'fold' + str(k+1)
        # make path
        model_save_path = model_save_path_base + '/' + fold_type
        funcs.Make_path(model_save_path)

        print('dataset: ', dataset)
        print('lr: ', lr)
        print('wd: ', wd)
        print('batch_size: ', b_size)
        print('n_hidden: ', n_hidden)

        load_base = dataset_base + dataset + "/" + fold_type
        train_P = np.loadtxt(load_base + "/train_P.csv", dtype=int, delimiter=",",skiprows=1)
        train_N = np.loadtxt(load_base + "/train_N.csv", dtype=int, delimiter=",",skiprows=1)
        dev_P = np.loadtxt(load_base + "/dev_P.csv", dtype=int, delimiter=",",skiprows=1)
        dev_N = np.loadtxt(load_base + "/dev_N.csv", dtype=int, delimiter=",",skiprows=1)

        test_P = np.loadtxt(load_base + "/test_P.csv", dtype=int, delimiter=",", skiprows=1)
        test_N = np.loadtxt(load_base + "/test_N.csv", dtype=int, delimiter=",", skiprows=1)

        print('number of DDI: ', len(train_P), len(dev_P), len(test_P))
        print('number of Negative DDI ', len(train_N), len(dev_N), len(test_N))

        # trans samples to id map and get X Y
        train_X, train_Y = Get_sample(train_P, train_N)
        dev_X, dev_Y = Get_sample(dev_P, dev_N)
        test_X, test_Y = Get_sample(test_P, test_N)
        # get loader

        train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
        dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)
        test_loader = funcs.get_test_loader(test_X, test_Y, b_size)

        for m in range(n_p_feats):
            for n in range(n_p_feats):
                n_dr_f = len(protein_features[m][0])
                n_dr2_f = len(protein_features[n][0])
                print('drug1 feature length: ', n_dr_f)
                print('drug2 feature length: ', n_dr2_f)
                model = DNNNet(n_dr_f, n_dr2_f, n_hidden).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                model_number = str(m * n_p_feats + n)
                print('model number: ', model_number)
                best_auc, best_epoch = 0, 0
                drug_feature = protein_features[m]
                protein_feature = protein_features[n]
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
                test_model = DNNNet(n_dr_f, n_dr2_f, n_hidden).to(device)
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
    for k in range(fold_number):
        fold_type = 'fold' + str(k+1)
        model_save_path = model_save_path_base + '/' + fold_type
        all_labels = np.loadtxt(model_save_path + '/test_labels.csv', skiprows=1)
        all_output_scores = []
        for i in range(n_p_feats):
            for j in range(n_p_feats):
                model_count = i * n_p_feats + j
                this_scores = np.loadtxt(model_save_path + '/test_scores' + str(model_count) + '.csv', skiprows=1)
                all_output_scores.append(this_scores)
        all_output_scores = list(np.mean(np.array(all_output_scores), axis=0))
        best_test = funcs.get_metric(all_labels, all_output_scores)
        for m in range(7):
            output_score[m][k] = best_test[m]
    mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision = show_metric(output_score,
                                                                                                dataset_base)

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
dataset_base = 'datasets_for_MDA/'

lr = 1e-3
wd = 1e-5
b_size = 128

n_hidden = 128
num_epoches = 200

losses = nn.BCELoss()


def Get_m_embedding():
    emb_feature_path_m = dataset_base + 'datasets/'
    m_fs = np.loadtxt(emb_feature_path_m + 'm_fs.csv', dtype=float, delimiter=',')
    m_gs = np.loadtxt(emb_feature_path_m + 'm_gs.csv', dtype=float, delimiter=',')
    m_ss = np.loadtxt(emb_feature_path_m + 'm_ss.csv', dtype=float, delimiter=',')
    m_embedding = {'m_fs': m_fs, 'm_gs': m_gs, 'm_ss': m_ss}
    m_embedding = data_loader.Trans_feature(m_embedding)
    M_features = [m_embedding['m_fs'], m_embedding['m_gs'], m_embedding['m_ss']]
    return M_features


def Get_d_embedding():
    emb_feature_path_d = dataset_base + 'datasets/'
    d_ts = np.loadtxt(emb_feature_path_d + 'd_ts.csv', dtype=float, delimiter=',')
    d_gs = np.loadtxt(emb_feature_path_d + 'd_gs.csv', dtype=float, delimiter=',')
    d_ss = np.loadtxt(emb_feature_path_d + 'd_ss.csv', dtype=float, delimiter=',')
    d_embedding = {'d_ts': d_ts, 'd_gs': d_gs, 'd_ss': d_ss}
    d_embedding = data_loader.Trans_feature(d_embedding)
    D_features = [d_embedding['d_ts'], d_embedding['d_gs'], d_embedding['d_ss']]
    return D_features


def Get_sample(DTI, N_DTI):
    P_list, N_list = [], []
    P_label, N_label = [], []
    for i in range(len(DTI)):
        P_list.append([int(DTI[i][0]), int(DTI[i][1])])
        P_label.append([1])
    for j in range(len(N_DTI)):
        N_list.append([int(N_DTI[j][0]), int(N_DTI[j][1])])
        N_label.append([0])
    X = np.concatenate((P_list, N_list))
    Y = np.concatenate((P_label, N_label))
    return X, Y

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

m_features = Get_m_embedding()
d_features = Get_d_embedding()
# dr_id_map, p_id_map, Drug_features, Protein_features = data_loader.Get_feature(dataset, input_type)
n_m_feats = len(m_features)
n_d_feats = len(d_features)
print('number of m feature types: ', n_m_feats)
print('number of d feature types: ', n_d_feats)

# make path
model_save_path_base = 'models_MDA'
funcs.Make_path(model_save_path_base)
# start
all_output_results = pd.DataFrame()

print('lr: ', lr)
print('wd: ', wd)
print('batch_size: ', b_size)
print('n_hidden: ', n_hidden)

# data load path
for k in range(5):
    fold_type = "fold" + str(k + 1)
    model_save_path = model_save_path_base + '/' + fold_type
    funcs.Make_path(model_save_path)
    train_P = np.loadtxt(dataset_base + "datasets/" + fold_type + "/train_pos.csv", dtype=int, delimiter=",",skiprows=1)
    train_N = np.loadtxt(dataset_base + "datasets/" + fold_type + "/train_neg.csv", dtype=int, delimiter=",",skiprows=1)
    dev_P = np.loadtxt(dataset_base + "datasets/" + fold_type + "/valid_pos.csv", dtype=int, delimiter=",", skiprows=1)
    dev_N = np.loadtxt(dataset_base + "datasets/" + fold_type + "/valid_neg.csv", dtype=int, delimiter=",", skiprows=1)

    test_P = np.loadtxt(dataset_base + "train_test/test_pos.csv", dtype=int, delimiter=",", skiprows=0)
    test_N = np.loadtxt(dataset_base + "train_test/test_neg.csv", dtype=int, delimiter=",", skiprows=0)
    print("number of MDA: ", len(train_P), len(dev_P), len(test_P))
    print("number of Negative MDA ", len(train_N), len(dev_N), len(test_N))

    # trans samples to id map and get X Y
    train_X, train_Y = Get_sample(train_P, train_N)
    dev_X, dev_Y = Get_sample(dev_P, dev_N)
    test_X, test_Y = Get_sample(test_P, test_N)
    # get loader
    train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
    dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)
    test_loader = funcs.get_test_loader(test_X, test_Y, b_size)

    for m in range(n_m_feats):
        for n in range(n_d_feats):
            n_m_f = len(m_features[m][0])
            n_d_f = len(d_features[n][0])
            print('m feature length: ', n_m_f)
            print('d feature length: ', n_d_f)
            model = DNNNet(n_m_f, n_d_f, n_hidden).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            model_number = str(m * n_d_feats + n)
            print('model number: ', model_number)
            best_auc, best_epoch = 0, 0
            drug_feature = m_features[m]
            protein_feature = d_features[n]
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
            test_model = DNNNet(n_m_f, n_d_f, n_hidden).to(device)
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
    for i in range(n_m_feats):
        for j in range(n_d_feats):
            model_count = i * n_d_feats + j
            this_scores = np.loadtxt(model_save_path + '/test_scores' + str(model_count) + '.csv', skiprows=1)
            all_output_scores.append(this_scores)
    all_output_scores = list(np.mean(np.array(all_output_scores), axis=0))
    best_test = funcs.get_metric(all_labels, all_output_scores)
    for m in range(7):
        output_score[m][k] = best_test[m]
mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision = show_metric(output_score, dataset_base)

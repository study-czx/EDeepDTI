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
dataset_base = 'datasets_for_DDI_before/'
datasets = ['zhang']
dataset_dict = {'deep': 'DeepDDI', 'miner': 'ChChMiner', 'zhang': 'ZhangDDI'}

lr = 1e-3
wd = 1e-5
b_size = 256

n_hidden = 128
num_epoches = 200

losses = nn.BCELoss()



def Get_drug_embedding(data_type):
    emb_feature_path_dr = dataset_base + 'feature/' + data_type + '_'
    chemberta = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2.csv', dtype=float, delimiter=',')
    grover_atom = np.loadtxt(emb_feature_path_dr + 'grover_atom.csv', dtype=float, delimiter=',')
    grover_bond = np.loadtxt(emb_feature_path_dr + 'grover_bond.csv', dtype=float, delimiter=',')
    molformer = np.loadtxt(emb_feature_path_dr + 'Molformer.csv', dtype=float, delimiter=',')
    # molclr = np.loadtxt(emb_feature_path_dr + 'molclr_emb.csv', dtype=float, delimiter=',')
    chemberta_max = np.loadtxt(emb_feature_path_dr + 'ChemBERTa2_max.csv', dtype=float, delimiter=',')
    grover_atom_max = np.loadtxt(emb_feature_path_dr + 'grover_atom_max.csv', dtype=float, delimiter=',')
    grover_bond_max = np.loadtxt(emb_feature_path_dr + 'grover_bond_max.csv', dtype=float, delimiter=',')
    molformer_max = np.loadtxt(emb_feature_path_dr + 'Molformer_max.csv', dtype=float, delimiter=',')
    # molclr_max = np.loadtxt(emb_feature_path_dr + 'molclr_emb_max.csv', dtype=float, delimiter=',')
    Dr_embedding = {'chemberta': chemberta, 'grover_atom': grover_atom, 'grover_bond': grover_bond,
                    'molformer': molformer, 'chemberta_max': chemberta_max, 'grover_atom_max': grover_atom_max,
                    'grover_bond_max': grover_bond_max, 'molformer_max': molformer_max}
    Dr_embedding = data_loader.Trans_feature(Dr_embedding)
    Drug_features = [Dr_embedding['chemberta'], Dr_embedding['grover_atom'], Dr_embedding['grover_bond'],
                     Dr_embedding['molformer'], Dr_embedding['chemberta_max'],
                     Dr_embedding['grover_atom_max'], Dr_embedding['grover_bond_max'],
                     Dr_embedding['molformer_max']]
    return Drug_features


def Get_sample(data, dr_id_map):
    data_list = []
    data_label = []
    for i in range(len(data)):
        data_list.append([dr_id_map[data['drugbank_id_1'][i]], dr_id_map[data['drugbank_id_2'][i]]])
        data_label.append([int(data['label'][i])])
    X = np.array(data_list)
    Y = np.array(data_label)
    return X, Y


for dataset in datasets:
    if dataset == 'deep' or dataset == 'zhang':
        b_size = 512
    else:
        b_size = 256
    Drug_id_data = pd.read_csv(dataset_base + 'drug_list_' + dataset + '.csv')
    Drug_id = Drug_id_data['drugbank_id']
    # get id map and features
    dr_id_map = funcs.id_map(Drug_id)
    drug_features = Get_drug_embedding(dataset)
    # dr_id_map, p_id_map, Drug_features, Protein_features = data_loader.Get_feature(dataset, input_type)
    n_dr_feats = len(drug_features)
    print('number of drug feature types: ', n_dr_feats)

    # make path
    model_save_path_base = 'models_DDI/' + dataset
    funcs.Make_path(model_save_path_base)
    # start
    all_output_results = pd.DataFrame()
    base_path = dataset_base + dataset
    print('dataset: ', dataset)
    print('lr: ', lr)
    print('wd: ', wd)
    print('batch_size: ', b_size)
    print('n_hidden: ', n_hidden)

    # model save path
    model_save_path = model_save_path_base
    funcs.Make_path(model_save_path)
    # data load path
    load_path = dataset_base + dataset_dict[dataset] + '_'

    train = pd.read_csv(load_path + 'train.csv')
    dev = pd.read_csv(load_path + 'valid.csv')
    test = pd.read_csv(load_path + 'test.csv')

    train_P, train_N = train[train['label']==1], train[train['label']==0]
    dev_P, dev_N = dev[dev['label'] == 1], dev[dev['label'] == 0]
    test_P, test_N = test[test['label'] == 1], test[test['label'] == 0]

    print('number of data: ', len(train), len(dev), len(test))
    print('number of DDI: ', len(train_P), len(dev_P), len(test_P))
    print('number of Negative DDI ', len(train_N), len(dev_N), len(test_N))


    # trans samples to id map and get X Y
    train_X, train_Y = Get_sample(train, dr_id_map)
    dev_X, dev_Y = Get_sample(dev, dr_id_map)
    test_X, test_Y = Get_sample(test, dr_id_map)
    # get loader
    train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
    dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)
    test_loader = funcs.get_test_loader(test_X, test_Y, b_size)

    for m in range(n_dr_feats):
        for n in range(n_dr_feats):
            n_dr_f = len(drug_features[m][0])
            n_dr2_f = len(drug_features[n][0])
            print('drug1 feature length: ', n_dr_f)
            print('drug2 feature length: ', n_dr2_f)
            model = DNNNet(n_dr_f, n_dr2_f, n_hidden).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            model_number = str(m * n_dr_feats + n)
            print('model number: ', model_number)
            best_auc, best_epoch = 0, 0
            drug_feature = drug_features[m]
            protein_feature = drug_features[n]
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

    all_labels = np.loadtxt(model_save_path + '/test_labels.csv', skiprows=1)
    all_output_scores = []
    for i in range(n_dr_feats):
        for j in range(n_dr_feats):
            model_count = i * n_dr_feats + j
            this_scores = np.loadtxt(model_save_path + '/test_scores' + str(model_count) + '.csv', skiprows=1)
            all_output_scores.append(this_scores)
    all_output_scores = np.array(all_output_scores)
    all_output_scores = np.mean(all_output_scores, axis=0)
    all_output_scores = list(all_output_scores)
    test_scores_label = funcs.computer_label(all_output_scores, 0.5)

    test_acc = skm.accuracy_score(all_labels, test_scores_label)
    test_auc = skm.roc_auc_score(all_labels, all_output_scores)
    test_aupr = skm.average_precision_score(all_labels, all_output_scores)
    test_mcc = skm.matthews_corrcoef(all_labels, test_scores_label)
    test_F1 = skm.f1_score(all_labels, test_scores_label)
    test_recall = skm.recall_score(all_labels, test_scores_label)
    test_precision = skm.precision_score(all_labels, test_scores_label)
    print(test_acc, test_auc, test_aupr, test_mcc, test_F1, test_recall, test_precision)
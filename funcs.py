import numpy as np
import random
import torch
from torch.utils import data
from pathlib import Path
import os
import sklearn.metrics as skm
import pandas as pd
import json

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def id_map(my_id):
    id_map = {"interger_id": "origin_id"}
    for i in range(len(my_id)):
        id_map[my_id[i]] = i
    return id_map

def Get_sample(DTI, N_DTI, dr_id_map, p_id_map):
    P_list, N_list = [],[]
    P_label, N_label = [],[]
    for i in range(len(DTI)):
        P_list.append([dr_id_map[DTI[i][0]], p_id_map[DTI[i][1]]])
        P_label.append([1])
    for j in range(len(N_DTI)):
        N_list.append([dr_id_map[N_DTI[j][0]], p_id_map[N_DTI[j][1]]])
        N_label.append([0])
    X = np.concatenate((P_list, N_list))
    Y = np.concatenate((P_label, N_label))
    return X, Y

def Get_Train_sample(DTI, N_DTI, dr_id_map, p_id_map):
    P_list, N_list = [],[]
    P_label, N_label = [],[]
    for i in range(len(DTI)):
        P_list.append([dr_id_map[DTI[i][0]], p_id_map[DTI[i][1]]])
        P_label.append([1])
    for j in range(len(N_DTI)):
        N_list.append([N_DTI[j][0], N_DTI[j][1]])
        N_label.append([0])
    X = np.concatenate((P_list, N_list))
    Y = np.concatenate((P_label, N_label))
    return X, Y

def Get_index(data, id_map1, id_map2):
    my_list = []
    for i in range(len(data)):
        my_list.append([id_map1[data[i][0]], id_map2[data[i][1]]])
    return my_list


def get_train_loader(X, Y, b_size):
    class Dataset(data.Dataset):
        def __init__(self):
            self.Data = X
            self.Label = Y

        def __getitem__(self, index):
            txt = torch.from_numpy(self.Data[index])
            label = torch.tensor(self.Label[index])
            return txt, label

        def __len__(self):
            return len(self.Data)

    Data = Dataset()
    loader = data.DataLoader(Data, batch_size=b_size, shuffle=True, drop_last=True, num_workers=0)
    return loader

def get_dev_loader(X, Y, b_size):
    class Dataset(data.Dataset):
        def __init__(self):
            self.Data = X
            self.Label = Y

        def __getitem__(self, index):
            txt = torch.from_numpy(self.Data[index])
            label = torch.tensor(self.Label[index])
            return txt, label

        def __len__(self):
            return len(self.Data)

    Data = Dataset()
    loader = data.DataLoader(Data, batch_size=b_size, shuffle=True, drop_last=True, num_workers=0)
    return loader

def get_test_loader(X, Y, b_size):
    class Dataset(data.Dataset):
        def __init__(self):
            self.Data = X
            self.Label = Y

        def __getitem__(self, index):
            txt = torch.from_numpy(self.Data[index])
            label = torch.tensor(self.Label[index])
            return txt, label

        def __len__(self):
            return len(self.Data)

    Data = Dataset()
    loader = data.DataLoader(Data, batch_size=b_size, shuffle=False, num_workers=0)
    return loader

def computer_label(input, threshold):
    label = []
    for i in range(len(input)):
        if (input[i] >= threshold):
            y = 1
        else:
            y = 0
        label.append(y)
    return label

def shuffer(X, Y ,seed):
    index = [i for i in range(len(X))]
    np.random.seed(seed)
    np.random.shuffle(index)
    new_X, new_Y = X[index], Y[index]
    return new_X, new_Y

def delete_smalle_sim(sim, remain_ratio):
    data = []
    for i in range(1, len(sim)):
        for j in range(0, i):
            data.append(sim[i][j])
    data.sort(reverse=True)
    number_remain = int(len(data)*remain_ratio)
    number_th = data[number_remain-1]
    sim[sim < number_th] = 0
    return sim


def Make_path(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        os.makedirs(data_path)

def get_metric(all_labels, all_output_scores):
    test_scores_label = computer_label(all_output_scores, 0.5)
    test_acc = skm.accuracy_score(all_labels, test_scores_label)
    test_auc = skm.roc_auc_score(all_labels, all_output_scores)
    test_aupr = skm.average_precision_score(all_labels, all_output_scores)
    test_mcc = skm.matthews_corrcoef(all_labels, test_scores_label)
    test_F1 = skm.f1_score(all_labels, test_scores_label)
    test_recall = skm.recall_score(all_labels, test_scores_label)
    test_precision = skm.precision_score(all_labels, test_scores_label)

    # print(test_acc, test_auc, test_aupr, test_mcc, test_F1)
    this_test_result = [format(test_acc, '.4f'), format(test_auc, '.4f'), format(test_aupr, '.4f'),
                 format(test_mcc, '.4f'), format(test_F1, '.4f'), format(test_recall, '.4f'),
                 format(test_precision, '.4f')]
    return this_test_result

def show_metric(output_score, result_path, input_type):
    # print(output_score)
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
    pd_output.to_csv(result_path + '_score_'+input_type+'.csv', index=False)
    return mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision

def get_result(model_save_path_base, n_dr_feats, n_p_feats, n_fold):
    output_score = np.zeros(shape=(7, 5))
    for k in range(n_fold):
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
        best_test = get_metric(all_labels, all_output_scores)
        for m in range(7):
            output_score[m][k] = best_test[m]
    output_score2 = pd.DataFrame(output_score).T
    output_score2.columns = ['ACC', 'AUC', 'AUPR', 'MCC', 'F1', 'Recall', 'Precision']
    pd_out = output_score2[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
    return pd_out

def get_list_result(drug_feature_list, protein_feature_list, model_save_path_base, n_dr_feats, n_p_feats, n_fold):
    output_score = np.zeros(shape=(7, 5))
    for k in range(n_fold):
        fold_type = 'fold' + str(k + 1)
        model_save_path = model_save_path_base + '/' + fold_type
        all_labels = np.loadtxt(model_save_path + '/test_labels.csv', skiprows=1)
        all_output_scores = []
        for i in drug_feature_list:
            for j in protein_feature_list:
                model_number = i * n_p_feats + j
                this_scores = np.loadtxt(model_save_path + '/test_scores' + str(model_number) + '.csv',skiprows=1)
                all_output_scores.append(this_scores)
        all_output_scores = list(np.mean(np.array(all_output_scores), axis=0))
        best_test = get_metric(all_labels, all_output_scores)
        for m in range(7):
            output_score[m][k] = best_test[m]
    output_score2 = pd.DataFrame(output_score).T
    output_score2.columns = ['ACC', 'AUC', 'AUPR', 'MCC', 'F1', 'Recall', 'Precision']
    pd_out = output_score2[['AUC', 'AUPR', 'ACC', 'MCC', 'F1']]
    return pd_out

def get_feature_name(index, feature_type, input_type):
    if feature_type == 'drug':
        if input_type == 'e':
            drug_names = {0: 'maccs', 1: 'pubchem', 2: 'ecfp4', 3: 'fcfp4',
                          4: 'chemberta2', 5: 'molformer', 6: 'grover', 7: 'kpgt',
                          8: 'maccs_sim', 9: 'pubchem_sim', 10: 'ecfp4_sim', 11: 'fcfp4_sim', 12: 'DDI_sim'}
        else:  # input_type == 'd'
            drug_names = {0: 'ecfp4', 1: 'fcfp4', 2: 'maccs', 3: 'pubchem'}
        return drug_names.get(index, f'drug_{index}')
    else: # protein
        if input_type == 'e':
            protein_names = {0: 'prottrans', 1: 'protein_bert', 2: 'esm2',
                             3: 'prottrans_max', 4: 'protein_bert_max', 5: 'esm2_max',
                             6: 'seq', 7: 'PPI_a', 8: 'PPI2', 9: 'MF', 10: 'BP', 11: 'CC'}
        else:
            protein_names = {0: 'prottrans', 1: 'protein_bert', 2: 'esm2',
                             3: 'prottrans_max', 4: 'protein_bert_max', 5: 'esm2_max'}
        return protein_names.get(index, f'protein_{index}')


def get_cross_validation_metric(drug_feature_list, protein_feature_list, model_save_path_base, n_p_feats, n_fold, data_type='val'):
    auc_list, aupr_list = [], []
    for k in range(n_fold):
        fold_type = 'fold' + str(k + 1)
        model_save_path = model_save_path_base + '/' + fold_type
        labels_file = model_save_path + '/' + data_type + '_labels.csv'
        if not os.path.exists(labels_file):
            continue
        all_labels = np.loadtxt(labels_file, skiprows=1)
        all_output_scores = []

        for i in drug_feature_list:
            for j in protein_feature_list:
                model_idx = i * n_p_feats + j
                scores_file = model_save_path + '/' + data_type + '_scores' + str(model_idx) + '.csv'
                if os.path.exists(scores_file):
                    this_scores = np.loadtxt(scores_file, skiprows=1)
                    all_output_scores.append(this_scores)
        if not all_output_scores:
            continue
        avg_scores = np.mean(np.array(all_output_scores), axis=0)
        test_auc = skm.roc_auc_score(all_labels, avg_scores)
        test_aupr = skm.average_precision_score(all_labels, avg_scores)
        auc_list.append(test_auc)
        aupr_list.append(test_aupr)

    if not auc_list:
        return 0, 0

    mean_auc = np.mean(auc_list)
    mean_aupr = np.mean(aupr_list)
    return mean_auc, mean_aupr


def greedy_forward_feature_selection_validation(model_save_path_base, n_dr_feats, n_p_feats, input_type, dataset, n_fold):
    metric_name = 'AUC' if dataset == 'DTI' else 'AUPR'

    # print(f"Dataset: {dataset}")
    # print(f"Drug features: {n_dr_feats}, Protein features: {n_p_feats}")
    # print(f"Greedy Forward Feature Selection Based on Validation {metric_name}")

    selected_drug_features = []
    selected_protein_features = []
    all_drug_features = list(range(n_dr_feats))
    all_protein_features = list(range(n_p_feats))

    best_val_metric = 0.0
    iteration = 0

    while True:
        iteration += 1
        best_candidate = None
        best_candidate_metric = 0.0
        candidate_type = None

        # Special handling for first selection (need to select both drug and protein features)
        if not selected_drug_features and not selected_protein_features:
            best_drug = None
            best_protein = None
            best_pair_metric = 0.0

            for drug_feat in all_drug_features:
                for protein_feat in all_protein_features:
                    val_auc, val_aupr = get_cross_validation_metric([drug_feat], [protein_feat],
                                                                    model_save_path_base, n_p_feats, n_fold, 'val')
                    val_metric = val_auc if dataset == 'DTI' else val_aupr

                    if val_metric >= best_pair_metric:
                        best_pair_metric = val_metric
                        best_drug = drug_feat
                        best_protein = protein_feat

            if best_drug is not None and best_protein is not None:
                selected_drug_features.append(best_drug)
                selected_protein_features.append(best_protein)
                best_val_metric = best_pair_metric
                continue
            else:
                print("Cannot find valid initial feature combination")
                break

        # Try adding each remaining drug feature
        remaining_drug_features = [f for f in all_drug_features if f not in selected_drug_features]
        for drug_feat in remaining_drug_features:
            temp_drug_features = selected_drug_features + [drug_feat]
            val_auc, val_aupr = get_cross_validation_metric(temp_drug_features, selected_protein_features,
                                                            model_save_path_base, n_p_feats, n_fold, 'val')
            val_metric = val_auc if dataset == 'DTI' else val_aupr

            if val_metric >= best_candidate_metric:
                best_candidate_metric = val_metric
                best_candidate = drug_feat
                candidate_type = 'drug'

        # Try adding each remaining protein feature
        remaining_protein_features = [f for f in all_protein_features if f not in selected_protein_features]
        for protein_feat in remaining_protein_features:
            temp_protein_features = selected_protein_features + [protein_feat]
            val_auc, val_aupr = get_cross_validation_metric(selected_drug_features, temp_protein_features,
                                                            model_save_path_base, n_p_feats, n_fold, 'val')
            val_metric = val_auc if dataset == 'DTI' else val_aupr

            if val_metric >= best_candidate_metric:
                best_candidate_metric = val_metric
                best_candidate = protein_feat
                candidate_type = 'protein'

        # If found a feature that improves performance
        if best_candidate_metric > best_val_metric:
            if candidate_type == 'drug':
                selected_drug_features.append(best_candidate)
                feature_name = get_feature_name(best_candidate, 'drug', input_type)
                # print(f"Selected drug feature: {feature_name}")
            else:
                selected_protein_features.append(best_candidate)
                feature_name = get_feature_name(best_candidate, 'protein', input_type)
                # print(f"Selected protein feature: {feature_name}")

            best_val_metric = best_candidate_metric
            # print(f"Current validation {metric_name}: {best_val_metric:.4f}")
            # print(
            #     f"Currently selected: {len(selected_drug_features)} drug features, {len(selected_protein_features)} protein features")
        else:
            # print(f"No feature found that improves {metric_name}, stopping selection")
            break

    # Display final selected features
    # print(f"\nFinally selected {len(selected_drug_features)} drug features and {len(selected_protein_features)} protein features")
    print(f"Total {len(selected_drug_features) * len(selected_protein_features)} base learners")
    print(f"Best validation {metric_name}: {best_val_metric:.4f}")

    print("\nSelected drug features:")
    for feat in selected_drug_features:
        print(f"- {get_feature_name(feat, 'drug', input_type)}")

    print("\nSelected protein features:")
    for feat in selected_protein_features:
        print(f"- {get_feature_name(feat, 'protein', input_type)}")

    # Save selected features to model path
    selected_features = {
        'dataset': dataset,
        'selection_metric': metric_name,
        'selected_drug_features': selected_drug_features,
        'selected_protein_features': selected_protein_features,
        'best_validation_metric': float(best_val_metric),
        'num_base_learners': len(selected_drug_features) * len(selected_protein_features),
        'feature_names': {
            'drug': [get_feature_name(f, 'drug', input_type) for f in selected_drug_features],
            'protein': [get_feature_name(f, 'protein', input_type) for f in selected_protein_features]
        }
    }

    # Save to model base path
    feature_file = model_save_path_base + '/selected_features_greedy.json'
    with open(feature_file, 'w') as f:
        json.dump(selected_features, f, indent=2)

    # print(f"\nSelected features saved to: {feature_file}")
    return selected_drug_features, selected_protein_features


def evaluate_greedy_selected_features_on_test(model_save_path_base, n_dr_feats, n_p_feats, result_save_path_base, n_fold):
    # Load saved feature selection results
    feature_file = model_save_path_base + '/selected_features_greedy.json'
    if not os.path.exists(feature_file):
        print(f"Selected features file not found: {feature_file}")
        return None

    with open(feature_file, 'r') as f:
        selected_features = json.load(f)

    selected_drug_features = selected_features['selected_drug_features']
    selected_protein_features = selected_features['selected_protein_features']

    # Calculate test set performance using existing get_list_result function
    test_results = get_list_result(selected_drug_features, selected_protein_features, result_save_path_base, n_dr_feats, n_p_feats, n_fold)
    return test_results

def greedy_backward_feature_elimination_validation(model_save_path_base, n_dr_feats, n_p_feats, input_type, dataset, n_fold):
    metric_name = 'AUC' if dataset == 'DTI' else 'AUPR'
    # Start with all features
    selected_drug_features = list(range(n_dr_feats))
    selected_protein_features = list(range(n_p_feats))

    # Calculate initial performance with all features
    val_auc, val_aupr = get_cross_validation_metric(selected_drug_features, selected_protein_features,
                                                    model_save_path_base, n_p_feats, n_fold,'val')
    best_val_metric = val_auc if dataset == 'DTI' else val_aupr

    iteration = 0

    while True:
        iteration += 1
        worst_feature = None
        worst_feature_type = None
        best_metric_after_removal = 0.0

        # Try removing each drug feature (only if more than 1 drug feature remains)
        if len(selected_drug_features) > 1:
            for drug_feat in selected_drug_features:
                temp_drug_features = [f for f in selected_drug_features if f != drug_feat]
                val_auc, val_aupr = get_cross_validation_metric(temp_drug_features, selected_protein_features,
                                                                model_save_path_base, n_p_feats, n_fold, 'val')
                val_metric = val_auc if dataset == 'DTI' else val_aupr

                if val_metric > best_metric_after_removal:
                    best_metric_after_removal = val_metric
                    worst_feature = drug_feat
                    worst_feature_type = 'drug'

        # Try removing each protein feature (only if more than 1 protein feature remains)
        if len(selected_protein_features) > 1:
            for protein_feat in selected_protein_features:
                temp_protein_features = [f for f in selected_protein_features if f != protein_feat]
                val_auc, val_aupr = get_cross_validation_metric(selected_drug_features, temp_protein_features,
                                                                model_save_path_base, n_p_feats, n_fold, 'val')
                val_metric = val_auc if dataset == 'DTI' else val_aupr

                if val_metric > best_metric_after_removal:
                    best_metric_after_removal = val_metric
                    worst_feature = protein_feat
                    worst_feature_type = 'protein'

        if worst_feature is not None and best_metric_after_removal >= best_val_metric:
            if worst_feature_type == 'drug':
                selected_drug_features.remove(worst_feature)
            else:
                selected_protein_features.remove(worst_feature)
            best_val_metric = best_metric_after_removal
        else:
            # print(f"No feature can be removed without significant performance drop, stopping elimination")
            break
        # Safety check: ensure we have at least one feature of each type
        if len(selected_drug_features) == 1 and len(selected_protein_features) == 1:
            print("Reached minimum feature set (1 drug, 1 protein), stopping elimination")
            break

    print("\nSelected drug features:")
    for feat in selected_drug_features:
        print(f"- {get_feature_name(feat, 'drug', input_type)}")

    print("\nSelected protein features:")
    for feat in selected_protein_features:
        print(f"- {get_feature_name(feat, 'protein', input_type)}")

    # Save selected features to model path
    selected_features = {
        'dataset': dataset,
        'selection_method': 'backward_elimination',
        'selection_metric': metric_name,
        'selected_drug_features': selected_drug_features,
        'selected_protein_features': selected_protein_features,
        'best_validation_metric': float(best_val_metric),
        'num_base_learners': len(selected_drug_features) * len(selected_protein_features),
        'feature_names': {
            'drug': [get_feature_name(f, 'drug', input_type) for f in selected_drug_features],
            'protein': [get_feature_name(f, 'protein', input_type) for f in selected_protein_features]
        }
    }

    # Save to model base path with different filename
    feature_file = model_save_path_base + '/selected_features_backward.json'
    with open(feature_file, 'w') as f:
        json.dump(selected_features, f, indent=2)

    print(f"\nSelected features saved to: {feature_file}")
    return selected_drug_features, selected_protein_features


def evaluate_backward_selected_features_on_test(model_save_path_base, n_dr_feats, n_p_feats, result_save_path_base, n_fold):
    # Load saved feature selection results
    feature_file = model_save_path_base + '/selected_features_backward.json'
    if not os.path.exists(feature_file):
        print(f"Selected features file not found: {feature_file}")
        return None

    with open(feature_file, 'r') as f:
        selected_features = json.load(f)

    selected_drug_features = selected_features['selected_drug_features']
    selected_protein_features = selected_features['selected_protein_features']

    # Calculate test set performance using existing get_list_result function
    test_results = get_list_result(selected_drug_features, selected_protein_features, result_save_path_base, n_dr_feats, n_p_feats, n_fold)
    return test_results
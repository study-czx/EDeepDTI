import numpy as np
import sklearn.metrics as skm
import funcs
import data_loader

dataset = 'DTI'
input_type = 'e'
# predict_types = ['5_fold', 'new_drug', 'new_protein', 'new_drug_protein']
predict_types = ['5_fold']
# predict_types = ['new_protein', 'new_drug_protein']
save_base = 'EDDTI'

save_base = save_base + '-' + input_type
# save_base = save_base

def Get_metric():
    n_dr_feats, n_p_feats = data_loader.Get_feature_numbers(dataset, input_type)
    if dataset == 'DTI' or dataset == 'CPI':
        for predict_type in predict_types:
            output_score = np.zeros(shape=(7, 5))
            for k in range(5):
                fold_type = 'fold' + str(k + 1)
                model_save_path = 'models/' + save_base + '/' + dataset + '/' + predict_type + '/' + fold_type
                all_labels = np.loadtxt(model_save_path + '/test_labels.csv', skiprows=1)
                all_output_scores = []
                for i in range(n_dr_feats):
                    for j in range(n_p_feats):
                        m = i * n_p_feats + j
                        this_scores = np.loadtxt(model_save_path + '/test_scores' + str(m) + '.csv', skiprows=1)
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

                print(test_acc, test_auc, test_aupr, test_mcc, test_F1)
                best_test = [format(test_acc, '.4f'), format(test_auc, '.4f'), format(test_aupr, '.4f'),
                             format(test_mcc, '.4f'), format(test_F1, '.4f'), format(test_recall, '.4f'),
                             format(test_precision, '.4f')]
                for m in range(7):
                    output_score[m][k] = best_test[m]
            # mean scores of 5 fold
            print(output_score)
            mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision = np.nanmean(
                output_score[0]), np.nanmean(output_score[1]), np.nanmean(output_score[2]), np.nanmean(
                output_score[3]), np.nanmean(output_score[4]), np.nanmean(output_score[5]), np.nanmean(output_score[6])
            std_acc, std_auc, std_aupr, std_mcc, std_f1, std_recall, std_precision = np.nanstd(output_score[0]), np.nanstd(
                output_score[1]), np.nanstd(output_score[2]), np.nanstd(output_score[3]), np.nanstd(
                output_score[4]), np.nanstd(output_score[5]), np.nanstd(output_score[6])
            print(mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision)
            print(std_acc, std_auc, std_aupr, std_mcc, std_f1, std_recall, std_precision)
    else:
        output_score = np.zeros(shape=(7, 5))
        for k in range(5):
            fold_type = 'fold' + str(k + 1)
            model_save_path = 'models/' + save_base + '/' + dataset + '/' + fold_type
            all_labels = np.loadtxt(model_save_path + '/test_labels.csv', skiprows=1)
            all_output_scores = []
            for i in range(n_dr_feats):
                for j in range(n_p_feats):
                    m = i * n_p_feats + j
                    this_scores = np.loadtxt(model_save_path + '/test_scores' + str(m) + '.csv', skiprows=1)
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

            print(test_acc, test_auc, test_aupr, test_mcc, test_F1)
            best_test = [format(test_acc, '.4f'), format(test_auc, '.4f'), format(test_aupr, '.4f'),
                         format(test_mcc, '.4f'), format(test_F1, '.4f'), format(test_recall, '.4f'),
                         format(test_precision, '.4f')]
            for m in range(7):
                output_score[m][k] = best_test[m]
        # mean scores of 5 fold
        print(output_score)
        mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision = np.nanmean(
            output_score[0]), np.nanmean(output_score[1]), np.nanmean(output_score[2]), np.nanmean(
            output_score[3]), np.nanmean(output_score[4]), np.nanmean(output_score[5]), np.nanmean(output_score[6])
        std_acc, std_auc, std_aupr, std_mcc, std_f1, std_recall, std_precision = np.nanstd(output_score[0]), np.nanstd(
            output_score[1]), np.nanstd(output_score[2]), np.nanstd(output_score[3]), np.nanstd(
            output_score[4]), np.nanstd(output_score[5]), np.nanstd(output_score[6])
        print(mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision)
        print(std_acc, std_auc, std_aupr, std_mcc, std_f1, std_recall, std_precision)

# Get_metric()

def Get_metric_all():
    n_dr_feats, n_p_feats = data_loader.Get_feature_numbers(dataset, input_type)
    for predict_type in predict_types:
        all_scores = []
        all_aupr_scores = []
        for i in range(n_dr_feats):
            for j in range(n_p_feats):
                m = i * n_p_feats + j
                print('药物序号为{}，蛋白序号为{}'.format(i+1,j+1))
                output_score = np.zeros(shape=(7, 5))
                for k in range(5):
                    fold_type = 'fold' + str(k + 1)
                    model_save_path = 'models/' + save_base + '/' + dataset + '/' + predict_type + '/' + fold_type
                    # print(model_save_path)
                    this_scores = np.loadtxt(model_save_path + '/test_scores' + str(m) + '.csv', skiprows=1)
                    all_labels = np.loadtxt(model_save_path + '/test_labels.csv', skiprows=1)

                    this_scores = list(np.array(this_scores))
                    test_scores_label = funcs.computer_label(this_scores, 0.5)

                    test_acc = skm.accuracy_score(all_labels, test_scores_label)
                    test_auc = skm.roc_auc_score(all_labels, this_scores)
                    test_aupr = skm.average_precision_score(all_labels, this_scores)
                    test_mcc = skm.matthews_corrcoef(all_labels, test_scores_label)
                    test_F1 = skm.f1_score(all_labels, test_scores_label)
                    test_recall = skm.recall_score(all_labels, test_scores_label)
                    test_precision = skm.precision_score(all_labels, test_scores_label)

                    # print(test_acc, test_auc, test_aupr, test_mcc, test_F1)
                    best_test = [format(test_acc, '.4f'), format(test_auc, '.4f'), format(test_aupr, '.4f'),
                                 format(test_mcc, '.4f'), format(test_F1, '.4f'), format(test_recall, '.4f'),
                                 format(test_precision, '.4f')]
                    for f in range(7):
                        output_score[f][k] = best_test[f]
                # mean scores of 5 fold
                print(output_score)
                mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision = np.nanmean(
                    output_score[0]), np.nanmean(output_score[1]), np.nanmean(output_score[2]), np.nanmean(
                    output_score[3]), np.nanmean(output_score[4]), np.nanmean(output_score[5]), np.nanmean(output_score[6])
                std_acc, std_auc, std_aupr, std_mcc, std_f1, std_recall, std_precision = np.nanstd(output_score[0]), np.nanstd(
                    output_score[1]), np.nanstd(output_score[2]), np.nanstd(output_score[3]), np.nanstd(
                    output_score[4]), np.nanstd(output_score[5]), np.nanstd(output_score[6])
                print(mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision)
                print(std_acc, std_auc, std_aupr, std_mcc, std_f1, std_recall, std_precision)
                all_scores.append(mean_auc)
                all_aupr_scores.append(mean_aupr)
        max_scores = np.max(all_scores)
        mean_scores = np.mean(all_scores)
        max_aupr_scores = np.max(all_aupr_scores)
        mean_aupr_scores = np.mean(all_aupr_scores)
        print('最大AUC值为： ', round(max_scores, 4))
        print('平均AUC值为： ', round(mean_scores, 4))
        print('最大AUPR值为： ', round(max_aupr_scores, 4))
        print('平均AUPR值为： ', round(mean_aupr_scores, 4))


Get_metric_all()

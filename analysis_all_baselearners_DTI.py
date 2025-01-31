import numpy as np
import sklearn.metrics as skm
import funcs
import data_loader
import pandas as pd

predict_type = '5_fold'


def get_results(model_id, type):
    auc_list = []
    for k in range(5):
        fold_type = 'fold' + str(k + 1)
        model_save_path = path_model_folder + save_base + '/' + dataset + '/' + predict_type + '/' + fold_type
        this_scores = np.loadtxt(model_save_path + '/' + type + '_scores' + str(model_id) + '.csv', skiprows=1)
        this_scores = list(np.array(this_scores))
        all_labels = np.loadtxt(model_save_path + '/' + type + '_labels.csv', skiprows=1)
        aucs = skm.roc_auc_score(all_labels, this_scores)
        auc_list.append(aucs)
    mean_auc = round(sum(auc_list) / len(auc_list), 4)
    return mean_auc

def Get_all_metric(drug_list, protein_list, type):
    p = 0
    all_auc_list = pd.DataFrame(columns=['drug', 'protein', 'auc'])
    all_auc_results = np.zeros((n_dr_feats, n_p_feats))
    for i in drug_list:
        for j in protein_list:
            m = i * n_p_feats + j
            this_auc = get_results(m, type)
            # print(this_auc)
            all_auc_results[i, j] = this_auc
            all_auc_list.loc[p, 'drug'] = i
            all_auc_list.loc[p, 'protein'] = j
            all_auc_list.loc[p, 'auc'] = this_auc
            p += 1
    # df_auc = pd.DataFrame(all_auc_results)
    # print(df_auc)
    # df_auc.columns = list(protein_embeddings.keys()) + list(protein_embeddings_max.keys()) + list(
    #     protein_descriptors.keys())
    # df_auc.index = list(drug_embeddings.keys()) + list(drug_embeddings_max.keys()) + list(
    #     drug_descriptors.keys())
    # df_out_auc = df_auc.copy()
    # df_out_auc['mean_rows'] = df_auc.mean(axis=1)
    # df_out_auc.loc['column_means'] = df_auc.mean(axis=0)
    # df_out_auc.to_csv('all_base_learners_aucs.csv')

    return all_auc_list



def Get_cross_metric(drug_feature_list, protein_feature_list, type):
    auc_list, aupr_list = [], []
    for k in range(5):
        fold_type = 'fold' + str(k + 1)
        model_save_path = path_model_folder + save_base + '/' + dataset + '/' + predict_type + '/' + fold_type
        all_labels = np.loadtxt(model_save_path + '/' + type + '_labels.csv', skiprows=1)
        all_output_scores = []

        for i in drug_feature_list:
            for j in protein_feature_list:
                m = i * n_p_feats + j
                this_scores = np.loadtxt(model_save_path + '/' + type + '_scores' + str(m) + '.csv', skiprows=1)
                all_output_scores.append(this_scores)
        all_output_scores = list(np.mean(np.array(all_output_scores), axis=0))
        test_auc = skm.roc_auc_score(all_labels, all_output_scores)
        test_aupr = skm.average_precision_score(all_labels, all_output_scores)
        auc_list.append(test_auc)
        aupr_list.append(test_aupr)
    mean_auc = round(sum(auc_list) / len(auc_list), 4)
    mean_aupr = round(sum(aupr_list) / len(aupr_list), 4)
    return mean_auc, mean_aupr


def Get_list_metric(drug_feature_list, protein_feature_list, type):
    auc_list, aupr_list = [], []
    for k in range(5):
        fold_type = 'fold' + str(k + 1)
        model_save_path = path_model_folder + save_base + '/' + dataset + '/' + predict_type + '/' + fold_type
        all_labels = np.loadtxt(model_save_path + '/' + type + '_labels.csv', skiprows=1)
        all_output_scores = []
        for length in range(len(drug_feature_list)):
            i = drug_feature_list[length]
            j = protein_feature_list[length]
            m = i * n_p_feats + j
            this_scores = np.loadtxt(model_save_path + '/' + type + '_scores' + str(m) + '.csv', skiprows=1)
            all_output_scores.append(this_scores)
        all_output_scores = list(np.mean(np.array(all_output_scores), axis=0))
        test_auc = skm.roc_auc_score(all_labels, all_output_scores)
        test_aupr = skm.average_precision_score(all_labels, all_output_scores)
        auc_list.append(test_auc)
        aupr_list.append(test_aupr)
    mean_auc = round(sum(auc_list) / len(auc_list), 4)
    mean_aupr = round(sum(aupr_list) / len(aupr_list), 4)
    return mean_auc


def Greedy_Get_metric(candidate_drug_features, candidate_protein_features, type):
    # 初始化
    current_combinations = []  # 当前选用的药物-蛋白特征组合
    best_score = 0  # 初始化最佳得分

    # 构造所有候选特征组合
    all_combinations = [
        (drug_feature, protein_feature)
        for drug_feature in candidate_drug_features
        for protein_feature in candidate_protein_features
    ]

    # 贪心算法主循环
    while len(all_combinations) > 0:
        found_improvement = False
        best_candidate = None

        # 遍历所有剩余的特征组合
        for combination in all_combinations:
            drug_feature, protein_feature = combination

            # 构造新的特征组合
            new_combinations = current_combinations + [combination]
            new_drug_features = [c[0] for c in new_combinations]
            new_protein_features = [c[1] for c in new_combinations]

            # 计算组合的得分
            score = Get_list_metric(new_drug_features, new_protein_features, type)

            # 如果得分提高，记录当前的最佳组合
            if score >= best_score:
                best_score = score
                best_candidate = combination
                found_improvement = True

        # 如果有改进，将最佳特征组合加入当前集合
        if found_improvement:
            current_combinations.append(best_candidate)
            all_combinations.remove(best_candidate)  # 从候选组合中移除已选择的组合

            print(f"Added combination {best_candidate} - New Best Score: {best_score:.4f}")
        else:
            # 如果没有改进，退出循环
            break

    # 输出最终结果
    final_drug_features = [c[0] for c in current_combinations]
    final_protein_features = [c[1] for c in current_combinations]

    print("\nOptimal Feature Set Found:")
    print("Drug Features:", final_drug_features)
    print("Protein Features:", final_protein_features)
    print("Best Score:", round(best_score, 4))
    return final_drug_features, final_protein_features


def Reverse_Greedy_Get_metric(candidate_drug_features, candidate_protein_features, type):
    # 初始化所有药物和蛋白特征组合
    current_combinations = [
        (drug_feature, protein_feature)
        for drug_feature in candidate_drug_features
        for protein_feature in candidate_protein_features
    ]
    best_score = Get_list_metric([comb[0] for comb in current_combinations], [comb[1] for comb in current_combinations], type)  # 计算初始得分

    print(f"Initial Best Score: {best_score:.4f}")

    # 贪心算法主循环
    while len(current_combinations) > 0:
        found_improvement = False
        worst_combination = None

        # 遍历所有特征组合，尝试移除一个组合
        for combination in current_combinations:
            new_combinations = [comb for comb in current_combinations if comb != combination]  # 移除当前组合
            new_drug_features = [comb[0] for comb in new_combinations]
            new_protein_features = [comb[1] for comb in new_combinations]

            # 计算新组合的得分
            score = Get_list_metric(new_drug_features, new_protein_features, type)
            if score >= best_score:
                best_score = score
                worst_combination = combination
                found_improvement = True

        # 如果找到改进，将最差组合移除
        if found_improvement:
            current_combinations.remove(worst_combination)
            print(f"Removed combination {worst_combination} - New Best Score: {best_score:.4f}")
        else:
            # 如果没有改进，退出循环
            break

    # 输出最终结果
    final_drug_features = [comb[0] for comb in current_combinations]
    final_protein_features = [comb[1] for comb in current_combinations]

    print("\nOptimal Feature Set Found:")
    print("Drug Features:", final_drug_features)
    print("Protein Features:", final_protein_features)
    print("Best Score:", round(best_score, 4))
    return final_drug_features, final_protein_features


if __name__ == '__main__':
    dataset = 'DTI'
    input_type = 'e'
    predict_types = ['5_fold']

    save_base = 'EDDTI-e-all'
    path_model_folder = 'all_test_models/'
    n_dr_feats = 18
    n_p_feats = 15

    drug_embeddings = {'chemberta': 0, 'chemberta_mtr': 1, 'grover': 2, 'molformer': 3, 'molclr': 4, 'kpgt': 5, 'selformer': 6}
    drug_embeddings_max = {'chemberta_max': 7, 'chemberta_mtr_max': 8, 'grover_max': 9, 'molformer_max': 10, 'molclr_max': 11, 'kpgt_max': 12, 'selformer_max': 13}
    drug_descriptors = {'maccs': 14, 'pubchem': 15, 'ecfp4': 16, 'fcfp4': 17}

    protein_embeddings = {'esm2': 0, 'protein_bert': 1, 'prottrans': 2, 'tape': 3, 'ankh': 4}
    protein_embeddings_max = {'esm2_max': 5, 'protein_bert_max': 6, 'prottrans_max': 7, 'tape_max': 8, 'ankh_max': 9}

    protein_descriptors = {'PAAC': 10, 'KSCTriad': 11, 'TPC': 12, 'CKSAAP': 13, 'CTD': 14}

    # 获取所有基学习器的得分
    need_drug_lists = [0, 1, 2, 3, 5, 14, 15, 16, 17]
    need_protein_lists = [0, 1, 2, 5, 6, 7]
    # Get_all_metric(need_drug_lists, need_protein_lists, type='test')

    # 去除蛋白描述符，去除Grover的重复特征，去除selformer。
    # need_drug_lists = [0, 1, 4, 5, 7, 18, 19, 21, 22]
    # need_protein_lists = [0, 1, 2, 5, 6, 7]
    # need_drug_lists = [18, 19, 21, 22]
    # need_protein_lists = [10, 11, 12, 13, 14]
    val_result = Get_cross_metric(need_drug_lists, need_protein_lists, type='val')
    print(val_result)
    test_result = Get_cross_metric(need_drug_lists, need_protein_lists, type='test')
    print(test_result)

    # top_k = 20
    # all_auc_df = Get_all_metric(need_drug_lists, need_protein_lists, type='val')
    # top_k_aucs_index =  all_auc_df.sort_values(by='auc', ascending=False).head(top_k)
    # new_drug_list, new_protein_list = top_k_aucs_index['drug'].to_list(), top_k_aucs_index['protein'].to_list()
    # print(new_drug_list)
    # print(new_protein_list)
    # output_auc = Get_list_metric(new_drug_list, new_protein_list, type='test')
    # print(output_auc)

    # print('正向贪心')
    # useful_drug, useful_protein = Greedy_Get_metric(need_drug_lists, need_protein_lists, 'val')
    # print(useful_drug, useful_protein)
    # result = Get_list_metric(useful_drug, useful_protein, type='test')
    # print(result)
    #
    #
    # # print('反向贪心')
    # useful_drug, useful_protein = Reverse_Greedy_Get_metric(need_drug_lists, need_protein_lists, type='val')
    # result = Get_list_metric(useful_drug, useful_protein, type='test')
    # print(result)

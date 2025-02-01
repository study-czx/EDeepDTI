import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed

def id_map(my_id):
    id_map = {"interger_id": "origin_id"}
    for i in range(len(my_id)):
        id_map[my_id[i]] = i
    return id_map

def cal_CPI_finger_sim():
    all_compound_id = pd.read_csv('../datasets_DTI/datasets/CPI/all_compound_id.csv')
    print(all_compound_id)
    dr_id_map = id_map(all_compound_id.iloc[:, 0].tolist())

    compound_id = pd.read_csv('../datasets_DTI/datasets/CPI/compound_id.csv')
    print(compound_id)
    compound_id_map = [dr_id_map[k] for k in compound_id.iloc[:, 0].tolist()]
    # print(compound_id_map)

    finger_types = ['ECFP4', 'FCFP4', 'MACCS', 'PubChem']
    for finger_type in finger_types:
        finger_data = pd.read_csv('../datasets_DTI/datasets/CPI/drug_finger/'+finger_type+'.csv')
        this_finger_data = finger_data.iloc[compound_id_map, :]
        print(this_finger_data)
        this_finger_list = np.array(this_finger_data)
        print('cal Start')
        finger_similarity = 1 - pairwise_distances(this_finger_list, metric='jaccard')
        np.fill_diagonal(finger_similarity, 1)
        print('cal OK')
        upper_triangular = finger_similarity[np.triu_indices_from(finger_similarity, k=1)]
        min_val = np.min(upper_triangular)
        max_val = np.max(upper_triangular)
        mean_val = np.mean(upper_triangular)
        median_val = np.median(upper_triangular)

        # 打印统计信息
        print(f"Min similarity: {min_val}")
        print(f"Max similarity: {max_val}")
        print(f"Mean similarity: {mean_val}")
        print(f"Median similarity: {median_val}")

        # 分布区间统计
        bins = np.linspace(0, 1, 11)  # 将 0-1 分为 10 个区间
        hist, bin_edges = np.histogram(upper_triangular, bins=bins)
        total_count = len(upper_triangular)
        proportions = hist / total_count
        # 打印区间统计结果
        for i in range(len(hist)):
            print(
                f"Interval [{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}): {hist[i]}, {round(proportions[i] * 100, 2)}%")
        # print(finger_similarity)
        # similarity_df = pd.DataFrame(finger_similarity).round(6)
        # print(similarity_df)
        # similarity_df.to_csv(output_path, index=False, header=False)
def cal_CPI_random_sim():
    finger_types = ['ECFP4', 'FCFP4', 'MACCS', 'PubChem']
    for finger_type in finger_types:
        finger_data = pd.read_csv('../datasets_DTI/datasets/CPI/drug_finger/'+finger_type+'.csv')
        this_finger_data = finger_data.sample(frac=0.05, random_state=42)
        print(this_finger_data)
        this_finger_list = np.array(this_finger_data)
        print('cal Start')
        finger_similarity = 1 - pairwise_distances(this_finger_list, metric='jaccard')
        np.fill_diagonal(finger_similarity, 1)
        print('cal OK')
        upper_triangular = finger_similarity[np.triu_indices_from(finger_similarity, k=1)]
        min_val = np.min(upper_triangular)
        max_val = np.max(upper_triangular)
        mean_val = np.mean(upper_triangular)
        median_val = np.median(upper_triangular)

        # 打印统计信息
        print(f"Min similarity: {min_val}")
        print(f"Max similarity: {max_val}")
        print(f"Mean similarity: {mean_val}")
        print(f"Median similarity: {median_val}")

        # 分布区间统计
        bins = np.linspace(0, 1, 11)  # 将 0-1 分为 10 个区间
        hist, bin_edges = np.histogram(upper_triangular, bins=bins)
        total_count = len(upper_triangular)
        proportions = hist / total_count
        # 打印区间统计结果
        for i in range(len(hist)):
            print(
                f"Interval [{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}): {hist[i]}, {round(proportions[i] * 100, 2)}%")
# cal_CPI_finger_sim()
cal_CPI_random_sim()

def cal_drug_sim():
    datasets = ['DTI', 'Davis_5fold', 'KIBA_5fold']
    base_path = '../datasets_DTI/datasets/'
    sim_types = ['ECFP4', 'FCFP4', 'MACCS', 'PubChem', 'DDI']

    for dataset in datasets:
        print(dataset)
        path = base_path + dataset + '/drug_sim/'
        if dataset != 'DTI':
            sim_types = ['ECFP4', 'FCFP4', 'MACCS', 'PubChem']
        for sim_type in sim_types:
            finger_sim = np.loadtxt(path + sim_type + '.csv', delimiter=',')
            # print(ECFP4_sim)

            upper_triangular = finger_sim[np.triu_indices_from(finger_sim, k=1)]
            min_val = np.min(upper_triangular)
            max_val = np.max(upper_triangular)
            mean_val = np.mean(upper_triangular)
            median_val = np.median(upper_triangular)

            # 打印统计信息
            print(f"Min similarity: {min_val}")
            print(f"Max similarity: {max_val}")
            print(f"Mean similarity: {mean_val}")
            print(f"Median similarity: {median_val}")

            # 分布区间统计
            bins = np.linspace(0, 1, 11)  # 将 0-1 分为 10 个区间
            hist, bin_edges = np.histogram(upper_triangular, bins=bins)
            total_count = len(upper_triangular)
            proportions = hist / total_count
            # 打印区间统计结果
            for i in range(len(hist)):
                print(f"Interval [{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}): {hist[i]}, {round(proportions[i]*100,2)}%")

# cal_drug_sim()

def cal_protein_sim():
    datasets = ['DTI', 'CPI', 'Davis_5fold', 'KIBA_5fold']
    base_path = '../datasets_DTI/datasets/'
    for dataset in datasets:
        print(dataset)
        path = base_path + dataset + '/protein_sim/'
        # print(ECFP4_sim)
        sim_types = ['seq', 'MF', 'BP', 'CC', 'PPI_a', 'PPI_t']
        for sim_type in sim_types:
            print(sim_type)
            seq_sim = np.loadtxt(path + sim_type + '.csv', delimiter=',', skiprows=1)
            upper_triangular = seq_sim[np.triu_indices_from(seq_sim, k=1)]
            min_val = np.min(upper_triangular)
            max_val = np.max(upper_triangular)
            mean_val = np.mean(upper_triangular)
            median_val = np.median(upper_triangular)

            # 打印统计信息
            # print(f"Min similarity: {min_val}")
            # print(f"Max similarity: {max_val}")
            print(f"Mean similarity: {mean_val}")
            # print(f"Median similarity: {median_val}")

            # 分布区间统计
            bins = np.linspace(0, 1, 11)  # 将 0-1 分为 10 个区间
            hist, bin_edges = np.histogram(upper_triangular, bins=bins)
            total_count = len(upper_triangular)
            proportions = hist / total_count
            # 打印区间统计结果
            # for i in range(len(hist)):
            #     print(f"Interval [{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}): {hist[i]}, {round(proportions[i]*100,2)}%")

# cal_protein_sim()
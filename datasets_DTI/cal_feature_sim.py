import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import warnings
import datasets_DTI.funcs as funcs
import networkx as nx
import math

warnings.filterwarnings('ignore')


def cal_finger_sim(data_types):
    finger_types = ['ECFP4', 'FCFP4', 'MACCS', 'PubChem', 'RDKit']
    # finger_types = ['ECFP4']
    for data_type in data_types:
        for finger_type in finger_types:
            data_path = 'datasets/' + data_type + '/drug_finger/' + finger_type + '.csv'
            output_path1 = 'datasets/' + data_type + '/drug_sim/'
            output_path = output_path1 + finger_type + '.csv'
            funcs.Make_path(output_path1)
            finger_data = pd.read_csv(data_path)
            # print(finger_data)
            finger_list = np.array(finger_data)
            finger_similarity = 1 - pairwise_distances(finger_list, metric='jaccard')
            np.fill_diagonal(finger_similarity, 1)
            # print(finger_similarity)
            similarity_df = pd.DataFrame(finger_similarity).round(6)
            # print(similarity_df)
            similarity_df.to_csv(output_path, index=False, header=False)


def cal_DDI_sim():
    DDI_data, _ = funcs.get_DDI_data()
    DDI_id1, DDI_id2 = DDI_data.iloc[:, 0].tolist(), DDI_data.iloc[:, 1].tolist()
    DDI_id = sorted(list(set(DDI_id1 + DDI_id2)))

    output_path1 = 'datasets/DTI/drug_sim/'
    output_path = output_path1 + 'DDI.csv'
    funcs.Make_path(output_path1)
    print(len(DDI_data))

    G = nx.Graph()
    print('add node')
    for i in range(len(DDI_id)):
        G.add_node(DDI_id[i], node_type='drug')
    print('add edge')
    for k in range(len(DDI_data)):
        G.add_edge(DDI_data.iloc[k, 0], DDI_data.iloc[k, 1], edge_type='interaction_with')

    DDI_adj_matrix = nx.adjacency_matrix(G)
    DDI_adj_matrix_dense = DDI_adj_matrix.toarray()
    print(DDI_adj_matrix_dense)
    DDI_adj_matrix_df = pd.DataFrame(DDI_adj_matrix_dense, columns=DDI_id, index=DDI_id)

    DTI_drug_id = pd.read_csv('datasets/DTI/Drug_id.csv')
    DTI_drug_id_need = DTI_drug_id.iloc[:, 0].tolist()
    DDI_need_matrix = DDI_adj_matrix_df.loc[DTI_drug_id_need]
    print(DDI_need_matrix)

    DDI_need_list = np.array(DDI_need_matrix)
    DDI_similarity = 1 - pairwise_distances(DDI_need_list, metric='jaccard')
    np.fill_diagonal(DDI_similarity, 1)
    # print(finger_similarity)
    similarity_df = pd.DataFrame(DDI_similarity).round(6)
    # print(similarity_df)
    similarity_df.to_csv(output_path, index=False, header=False)


def function_cal_PPI_sim_adj(G, PPI_id, protein_id):
    PPI_adj_matrix = nx.adjacency_matrix(G)
    PPI_adj_matrix_dense = PPI_adj_matrix.toarray()
    # print(PPI_adj_matrix_dense)
    PPI_adj_matrix_df = pd.DataFrame(PPI_adj_matrix_dense, columns=PPI_id, index=PPI_id)

    PPI_protein_id_need = protein_id.iloc[:, 0].tolist()
    # print(PPI_protein_id_need)
    PPI_need_matrix = PPI_adj_matrix_df.loc[PPI_protein_id_need]
    # print(PPI_need_matrix)

    PPI_need_list = np.array(PPI_need_matrix)
    PPI_similarity = 1 - pairwise_distances(PPI_need_list, metric='jaccard')
    np.fill_diagonal(PPI_similarity, 1)
    # print(finger_similarity)
    similarity_df = pd.DataFrame(PPI_similarity).round(6)
    return similarity_df
    # print(similarity_df)


def function_cal_PPI_sim_top(G, protein_id):
    shortest_path_length = [[0 for j in range(len(protein_id))] for i in range(len(protein_id))]
    PPI_sim_top = [[0 for j in range(len(protein_id))] for i in range(len(protein_id))]
    for i in range(len(protein_id)):
        for j in range(len(protein_id)):
            if nx.has_path(G, protein_id.iloc[i, 0], protein_id.iloc[j, 0]) == True:
                shortest_path_length[i][j] = nx.shortest_path_length(G, protein_id.iloc[i, 0], protein_id.iloc[j, 0],
                                                                     weight=None, method='dijkstra')
            else:
                shortest_path_length[i][j] = 0
    for i in range(len(protein_id)):
        for j in range(len(protein_id)):
            if i == j:
                PPI_sim_top[i][j] = 1
            elif shortest_path_length[i][j] != 0:
                PPI_sim_top[i][j] = 0.9 * math.exp(-shortest_path_length[i][j])
            else:
                PPI_sim_top[i][j] = 0
    PPI_sim_top_df = pd.DataFrame(PPI_sim_top).round(6)
    return PPI_sim_top_df


def cal_PPI_sim(data_types):
    PPI_data, PPI_id = funcs.get_PPI_data()
    PPI_id = sorted(PPI_id.iloc[:, 0].tolist())
    # print(PPI_id)
    print(len(PPI_data))

    G = nx.Graph()
    print('add node')
    for i in range(len(PPI_id)):
        G.add_node(PPI_id[i], node_type='protein')
    print('add edge')
    for k in range(len(PPI_data)):
        G.add_edge(PPI_data.iloc[k, 0], PPI_data.iloc[k, 1], edge_type='interaction_with')
    for type in data_types:
        output_path1 = 'datasets/' + type + '/protein_sim/'
        output_path_sim1 = output_path1 + 'PPI_a.csv'
        output_path_sim2 = output_path1 + 'PPI_t.csv'
        funcs.Make_path(output_path1)
        if type == 'DTI' or type == 'CPI':
            if type == 'DTI':
                protein_ids = pd.read_csv('datasets/DTI/Protein_id.csv')
            elif type == 'CPI':
                protein_ids = pd.read_csv('datasets/CPI/all_protein_id.csv')
            PPI_sim_adj = function_cal_PPI_sim_adj(G, PPI_id, protein_ids)
            PPI_sim_adj.to_csv(output_path_sim1, index=False, header=False)

            PPI_sim_top = function_cal_PPI_sim_top(G, protein_ids)
            PPI_sim_top.to_csv(output_path_sim2, index=False, header=False)
        else:
            if type == 'Davis_5fold':
                protein_ids = pd.read_csv('datasets/Davis_5fold/Uniprot_id.csv')
            elif type == 'KIBA_5fold':
                protein_ids = pd.read_csv('datasets/KIBA_5fold/Protein.csv')
            protein_list = protein_ids.iloc[:, 0]
            All_PPI_sim_adj = [[0 for j in range(len(protein_list))] for i in range(len(protein_list))]
            All_PPI_sim_top = [[0 for j in range(len(protein_list))] for i in range(len(protein_list))]
            inter_proteins = set(PPI_id) & set(protein_list)
            pd_inter_proteins = pd.DataFrame(inter_proteins)
            PPI_sim_adj = function_cal_PPI_sim_adj(G, PPI_id, pd_inter_proteins)
            PPI_sim_adj.columns = list(inter_proteins)
            PPI_sim_adj.index = list(inter_proteins)
            print(PPI_sim_adj)
            PPI_sim_top = function_cal_PPI_sim_top(G, pd_inter_proteins)
            PPI_sim_top.columns = list(inter_proteins)
            PPI_sim_top.index = list(inter_proteins)
            print(PPI_sim_top)
            for i in range(len(protein_list)):
                protein1 = protein_list[i]
                for j in range(len(protein_list)):
                    protein2 = protein_list[j]
                    if i == j:
                        All_PPI_sim_adj[i][j] = 1
                        All_PPI_sim_top[i][j] = 1
                    else:
                        if protein1 in list(inter_proteins) and protein2 in list(inter_proteins):
                            All_PPI_sim_adj[i][j] = PPI_sim_adj.loc[protein1, protein2]
                            All_PPI_sim_top[i][j] = PPI_sim_top.loc[protein1, protein2]
                        else:
                            All_PPI_sim_adj[i][j] = 0
                            All_PPI_sim_top[i][j] = 0
            All_PPI_sim_adj, All_PPI_sim_top = pd.DataFrame(All_PPI_sim_adj), pd.DataFrame(All_PPI_sim_top)
            All_PPI_sim_adj.to_csv(output_path_sim1, index=False, header=False)
            All_PPI_sim_top.to_csv(output_path_sim2, index=False, header=False)

data_types = ['DTI', 'Davis_5fold', 'KIBA_5fold']
# 1. cal finger sim of drugs
cal_finger_sim(data_types)

# 2. cal DDI sim of drugs (only for DTI dataset)
cal_DDI_sim()

data_types = ['DTI', 'CPI', 'Davis_5fold', 'KIBA_5fold']
# 3. cal PPI sim of proteins
cal_PPI_sim(data_types)

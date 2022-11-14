import numpy as np
import random
import torch
from torch.utils import data
import dgl

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)

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

def Get_weight_sim_graph(sim, num_node):
    sim_src, sim_dst, sim_value = [], [], []
    for i in range(len(sim)):
        for j in range(len(sim)):
            if sim[i][j] != 0:
                sim_src.append(i)
                sim_dst.append(j)
                sim_value.append(float(sim[i][j]))
    src, dst = torch.tensor(sim_src), torch.tensor(sim_dst)
    weight = torch.tensor(sim_value)
    Graph = dgl.graph((src, dst), num_nodes=num_node)
    return Graph, weight
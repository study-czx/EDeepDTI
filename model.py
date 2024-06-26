import torch
import torch.nn as nn
import torch.nn.functional as F

class DNNNet(nn.Module):
    def __init__(self, n_dr_f, n_protein_f, n_hidden):
        super(DNNNet, self).__init__()
        self.drug_hidden_layer = nn.Sequential(nn.Linear(in_features=n_dr_f, out_features=n_hidden), nn.ReLU())
        self.protein_hidden_layer = nn.Sequential(nn.Linear(in_features=n_protein_f, out_features=n_hidden), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(n_hidden * 2, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Drug_feature, Protein_feature, x_dr, x_p):
        dr_feat, p_feat = Drug_feature[x_dr].squeeze(1), Protein_feature[x_p].squeeze(1)
        h_dr = self.drug_hidden_layer(dr_feat)
        h_p = self.protein_hidden_layer(p_feat)
        h_dr_d = torch.cat((h_dr, h_p), dim=1)
        h_hidden = self.fc3(self.fc2(self.fc1(h_dr_d)))
        out = self.sigmoid(self.output(h_hidden))
        return out

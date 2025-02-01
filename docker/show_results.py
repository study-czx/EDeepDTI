import numpy as np
import pandas as pd
import os
from pathlib import Path

def Make_path(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        os.makedirs(data_path)


def read_vina_results(filename):
    model1_info = None
    affinities = []  # 存储前 5 个模式的结合能
    with open(filename, 'r') as file:
        lines = file.readlines()
        start_reading = False  # 用于标记何时开始读取模式数据

        for line in lines:
            # 识别表头行，下一行开始读取数据
            if line.strip().startswith("-----+------------+----------+----------"):
                start_reading = True
                continue

            if start_reading:
                parts = line.split()
                if len(parts) >= 4:  # 确保数据格式正确
                    mode = int(parts[0])
                    affinity = float(parts[1])
                    rmsd_lb = float(parts[2])
                    rmsd_ub = float(parts[3])

                    # 记录第一个模式的信息
                    if mode == 1:
                        model1_info = {
                            'mode': mode,
                            'affinity': affinity,
                            'rmsd_lb': rmsd_lb,
                            'rmsd_ub': rmsd_ub
                        }

                    # 记录前 5 个模式的结合能
                    if mode <= 5:
                        affinities.append(affinity)

    # 计算前 5 个模式的平均结合能
    mean_affinity_top5 = sum(affinities) / len(affinities) if affinities else None

    return model1_info, mean_affinity_top5

def get_protein_results(protein_pdb_name, drug_ids):
    record_pd = pd.DataFrame()
    for drug_id in drug_ids:
        # print(drug_id)
        result_log = 'result_log/' + drug_id + '_' + protein_pdb_name + '.txt'
        info, mean_affinity_top5 = read_vina_results(result_log)
        best_affinity = info['affinity']
        this_record = {'Uniprot_id': Uniprot_id, 'DrugBank_id': drug_id, 'Affinity': best_affinity, 'Mean_5_poses': mean_affinity_top5}
        pd_record = pd.DataFrame([this_record])
        if record_pd.empty:
            record_pd = pd_record
        else:
            record_pd = pd.concat([record_pd, pd_record])
    # print(record_pd)
    return record_pd



if __name__ == '__main__':
    Uniprot_ids = ['P00533', 'P53350']
    Uniprot_PDB_map = {'P00533': '3poz', 'P53350': '4o6w'}
    Make_path('affinity/')
    for Uniprot_id in Uniprot_ids:
        print(Uniprot_id)
        protein_pdb_name = Uniprot_PDB_map[Uniprot_id]
        print(protein_pdb_name)
        base_path = 'dock_pairs/'
        cal_pairs1 = pd.read_csv(base_path + 'pairs_top_end_' + Uniprot_id + '.csv')
        cal_pairs2 = pd.read_csv(base_path + 'pairs_DTI_' + Uniprot_id + '.csv')
        drug_ids1 = cal_pairs1['Drugbank_id'].to_list()
        drug_ids_top = drug_ids1[0:10]
        drug_ids_end = drug_ids1[10:20]
        drug_ids_DTI = cal_pairs2['Drugbank_id'].to_list()
        results_top = get_protein_results(protein_pdb_name, drug_ids_top)
        results_end = get_protein_results(protein_pdb_name, drug_ids_end)
        results_DTI = get_protein_results(protein_pdb_name, drug_ids_DTI)
        results_top.to_csv('affinity/top10_' + Uniprot_id + '.csv')
        results_end.to_csv('affinity/end10_' + Uniprot_id + '.csv')
        results_DTI.to_csv('affinity/DTI_' + Uniprot_id + '.csv')
        # print(results_top)
        # print(results_end)
        print(results_DTI)
        print(np.mean(results_top['Affinity'].to_list()), np.mean(results_end['Affinity'].to_list()), np.mean(results_DTI['Affinity'].to_list()))


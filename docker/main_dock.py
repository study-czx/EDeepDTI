import os
from pathlib import Path
import pandas as pd


def Make_path(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        os.makedirs(data_path)


def autodock(protein_pdb_name, drug_ids):
    save_result_path = 'result_pdbqt'
    save_log_path = 'result_log'
    Make_path(save_result_path)
    Make_path(save_log_path)

    for drug_id in drug_ids:
        print(drug_id)
        os.system(f'vina --ligand drug_pdbqt/{drug_id}.pdbqt  --config {protein_pdb_name}_conf.txt '
                  f'--out {save_result_path}/{drug_id}_{protein_pdb_name}.pdbqt --seed 12345'
                  f'> {save_log_path}/{drug_id}_{protein_pdb_name}.txt 2>&1'
                  )

def fix_pdbqt_out(Uniprot_ids):
    save_result_path = 'result_pdbqt'
    output_filename_path = 'result_pdbqt_fixed'
    Make_path(output_filename_path)
    for Uniprot_id in Uniprot_ids:
        print(Uniprot_id)
        cal_pairs1 = pd.read_csv(base_path + 'pairs_top_end_' + Uniprot_id + '.csv')
        cal_pairs2 = pd.read_csv(base_path + 'pairs_DTI_' + Uniprot_id + '.csv')
        protein_pdb_name = Uniprot_PDB_map[Uniprot_id]
        print(protein_pdb_name)
        drug_ids = cal_pairs1['Drugbank_id'].to_list() + cal_pairs2['Drugbank_id'].to_list()
        for drug_id in drug_ids:
            result_pdbqt = f'{save_result_path}/{drug_id}_{protein_pdb_name}.pdbqt'
            output_filename = f'{output_filename_path}/{drug_id}_{protein_pdb_name}.pdbqt'

            with open(result_pdbqt, 'r') as file:
                lines = file.readlines()

            indices_to_remove = []
            for i in range(1, len(lines)):
                if 'ENDMDL' in lines[i]:
                    indices_to_remove.append(i - 1)


            new_lines = [line for index, line in enumerate(lines) if index not in indices_to_remove]
            with open(output_filename, 'w') as file:
                file.writelines(new_lines)



if __name__ == '__main__':
    base_path = 'dock_pairs/'
    Uniprot_ids = ['O75469', 'P35968', 'P07949', 'Q00534', 'P29317', 'Q04912']
    Uniprot_PDB_map = {'O75469': '7axe', 'P35968': '2xir', 'P07949': '4ckj', 'Q00534': '6oqo', 'P29317': '6q7d',
                       'Q04912': '3pls'}


    for Uniprot_id in Uniprot_ids:
        print(Uniprot_id)
        cal_pairs1 = pd.read_csv(base_path + 'pairs_top_end_' + Uniprot_id + '.csv')
        cal_pairs2 = pd.read_csv(base_path + 'pairs_DTI_' + Uniprot_id + '.csv')
        protein_pdb_name = Uniprot_PDB_map[Uniprot_id]
        print(protein_pdb_name)
        drug_ids = cal_pairs1['Drugbank_id'].to_list() + cal_pairs2['Drugbank_id'].to_list()
        autodock(protein_pdb_name, drug_ids)

    # 删除pdbqt文件中的空行并保存到新文件夹中
    fix_pdbqt_out(Uniprot_ids)


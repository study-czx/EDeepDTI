import pandas as pd
import os
from pathlib import Path
import rdkit
import meeko
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from meeko import PDBQTWriterLegacy

def Make_path(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        os.makedirs(data_path)


all_drug_structure = pd.read_csv('../datasets_DTI/datasets/DTI/drug_smiles.csv', dtype='str')

output_dir = 'drug_pdbqt/'
Make_path(output_dir)

useful_ids = []

for i in range(len(all_drug_structure)):
    drugbank_id = all_drug_structure['Drugbank_id'][i]
    print(drugbank_id)
    this_smiles = all_drug_structure['SMILES'][i]
    lig = rdkit.Chem.MolFromSmiles(this_smiles)

    # 添加氢原子
    mol = Chem.AddHs(lig)
    # 生成3D构象
    etkdgv3 = rdDistGeom.srETKDGv3()

    if rdDistGeom.EmbedMolecule(mol, etkdgv3) == -1:
        print(f"Embedding failed for molecule {i + 1}")
        continue
    # 优化分子结构
    AllChem.UFFOptimizeMolecule(mol)

    # prepare for autodock
    meeko_prep = meeko.MoleculePreparation()
    mol_setup = meeko_prep.prepare(mol)[0]

    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setup)
    if is_ok:
        print(pdbqt_string, end="")
        useful_ids.append(drugbank_id)
    else:
        continue

    output_pdbqt = os.path.join(output_dir, f"{drugbank_id}.pdbqt")
    with open(output_pdbqt, 'w') as f:
        f.write(pdbqt_string)

useful_ids_pd = pd.DataFrame(useful_ids,columns=['Drugbank_id'])
useful_ids_pd.to_csv('DrugBank_ID_with_3D.csv', index=False)

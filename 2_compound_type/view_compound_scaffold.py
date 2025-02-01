from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

types = ['DTI', 'CPI', 'Davis_5fold', 'KIBA_5fold']
path_name_dict = {'DTI': 'drug_smiles', 'CPI': 'all_compound_smiles', 'Davis_5fold': 'Drug', 'KIBA_5fold': 'Drug'}
smiles_name_dict = {'DTI': 'SMILES', 'CPI': 'CanonicalSMILES', 'Davis_5fold': 'smiles', 'KIBA_5fold': 'smiles'}

for type in types:
    print(type)
    smiles_df = pd.read_csv('../datasets_DTI/datasets/' + type + '/' + path_name_dict[type] + '.csv')
    smiles_list = smiles_df[smiles_name_dict[type]].to_list()

    # 提取骨架并统计频率
    scaffolds = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            if scaffold_smiles:  # 确保骨架不是空字符串
                scaffolds.append(scaffold_smiles)

    # 统计骨架频率
    scaffold_counts = Counter(scaffolds)
    # 转换为 DataFrame
    df_scaffold = pd.DataFrame(scaffold_counts.items(), columns=["Scaffold", "Count"])
    df_scaffold = df_scaffold.sort_values(by="Count", ascending=False)

    unique_scaffolds = set(scaffolds)
    print(f"Unique scaffolds: {len(unique_scaffolds)}")

    # 输出常见骨架
    print(df_scaffold.head(10))  # 输出前 10 个常见骨架

    # 可视化
    # plt.figure(figsize=(10, 6))
    # plt.bar(df_scaffold["Scaffold"].head(10), df_scaffold["Count"].head(10))
    # plt.xlabel("Scaffold")
    # plt.ylabel("Frequency")
    # plt.title("Top 10 Most Common Scaffolds")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()
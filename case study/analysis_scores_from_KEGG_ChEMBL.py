import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get all scores
All_scores = pd.read_csv('All_scores_10fold_e.csv')
Drug_id = pd.read_csv('../datasets_DTI/datasets/DTI/Drug_id.csv')
Protein_id = pd.read_csv('../datasets_DTI/datasets/DTI/Protein_id.csv')
All_scores.index = Drug_id['Drugbank_id'].values.tolist()
All_scores.columns = Protein_id['Uniprot_id'].values.tolist()
# print(All_scores)

DTI_P = pd.read_csv('../datasets_DTI/datasets/DTI/DTI_P.csv')
DTI_N = pd.read_csv('../datasets_DTI/datasets/DTI/DTI_N.csv')

def anti_join(data1, data2):
    data_new = pd.merge(data1, data2, indicator=True, how='outer').query(
        '_merge=="left_only"').drop('_merge', axis=1).reset_index(drop=True)
    return data_new


def get_mean_of_P_N():
    scores_P = []
    scores_N = []

    for i in range(len(DTI_P)):
        this_drug = DTI_P.iloc[i, 0]
        this_protein = DTI_P.iloc[i, 1]
        this_score = All_scores.loc[this_drug, this_protein]
        scores_P.append(this_score)

    for i in range(len(DTI_N)):
        this_drug = DTI_N.iloc[i, 0]
        this_protein = DTI_N.iloc[i, 1]
        this_score = All_scores.loc[this_drug, this_protein]
        scores_N.append(this_score)

    print('mean score of Positive samples: ', np.mean(scores_P))
    print('mean score of Negative samples: ', np.mean(scores_N))


def get_mean_of_other_DTI(DTI_other):
    print('数据数量：', len(DTI_other))
    DTI_other_scores = []
    for i in range(len(DTI_other)):
        this_drug = DTI_other.iloc[i, 0]
        this_protein = DTI_other.iloc[i, 1]
        this_score = All_scores.loc[this_drug, this_protein]
        DTI_other_scores.append(this_score)
    # print(sorted(DTI_other_scores))
    print('mean score of DTI samples: ', np.mean(DTI_other_scores))
    bins = np.linspace(0, 1, 11)
    counts, bin_edges = np.histogram(DTI_other_scores, bins=bins)
    for i in range(len(counts)):
        percentage = (counts[i] / len(DTI_other)) * 100
        if i < len(counts) - 1:
            print(f"Interval [{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}): {counts[i]}, {percentage:.2f}%")
        else:
            print(f"Interval [{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}]: {counts[i]}, {percentage:.2f}%")

    # 使用matplotlib绘制直方图
    plt.hist(DTI_other_scores, bins=bins, edgecolor='black', align='left', rwidth=0.9)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Values in 10 Intervals')
    plt.show()


def analysis_DTI_scores():
    all_drug_protein_pairs = []
    for i in range(len(Drug_id)):
        for j in range(len(Protein_id)):
            this_drug = Drug_id['Drugbank_id'][i]
            this_protein = Protein_id['Uniprot_id'][j]
            all_drug_protein_pairs.append([this_drug, this_protein])
    all_dr_p_df = pd.DataFrame(all_drug_protein_pairs)
    print(all_dr_p_df)
    print(len(all_dr_p_df))
    all_dr_p_df.columns = ['Drugbank_id', 'Uniprot_id']
    all_dr_p_df1 = anti_join(all_dr_p_df, DTI_P)
    all_dr_p_df2 = anti_join(all_dr_p_df1, DTI_N)
    print(len(all_dr_p_df2))
    print('all drug-protein pairs out of training set')
    get_mean_of_other_DTI(all_dr_p_df2)

    KEGG_DTI = pd.read_csv('KEGG_DTI_match.csv')
    ChEMBL_DTI = pd.read_csv('ChEMBL_DTI_match.csv')
    print('KEGG database: ')
    get_mean_of_other_DTI(KEGG_DTI)
    print('ChEMBL database: ')
    get_mean_of_other_DTI(ChEMBL_DTI)


analysis_DTI_scores()
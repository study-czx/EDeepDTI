import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get all scores
All_scores = pd.read_csv('All_scores_10fold_e.csv', header=None)
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


def get_mean_of_other_DTI(DTI_other, th_score, name):
    print('数据数量：', len(DTI_other))
    DTI_other_scores = []
    for i in range(len(DTI_other)):
        this_drug = DTI_other.iloc[i, 0]
        this_protein = DTI_other.iloc[i, 1]
        this_score = All_scores.loc[this_drug, this_protein]
        DTI_other_scores.append(this_score)

    print('mean score of samples: ', np.mean(DTI_other_scores))

    count = len(list(filter(lambda x: x >= th_score, DTI_other_scores)))
    print('score > {} of DTI samples： {}'.format(th_score, count))
    count2 = len(list(filter(lambda x: x < 0.5, DTI_other_scores)))
    print('score < 0.5 of DTI samples： {}'.format(count2))
    count3 = len(list(filter(lambda x: x >= 0.5, DTI_other_scores)))
    print('score >= 0.5 of DTI samples： {}'.format(count3))

    bins = np.linspace(0, 1, 11)
    counts, bin_edges = np.histogram(DTI_other_scores, bins=bins)

    intervals = []
    counts_list = []
    percentages_list = []

    for i in range(len(counts)):
        percentage = round((counts[i] / len(DTI_other)) * 100, 2)
        intervals.append(f"[{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f})" if i < len(counts) - 1 else f"[{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}]")
        counts_list.append(counts[i])
        percentages_list.append(str(percentage)+'%')
    result_df = pd.DataFrame({
        'Interval': intervals,
        'Count_' + name: counts_list,
        'Percentage_' + name: percentages_list
    })
    print(result_df)
    return result_df

    # 使用matplotlib绘制直方图
    # plt.hist(DTI_other_scores, bins=bins, edgecolor='black', align='left', rwidth=0.9)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Values in 10 Intervals')
    # plt.show()


def analysis_all_DTI_scores(th_score):
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
    all_samples = get_mean_of_other_DTI(all_dr_p_df2, th_score, name='all')
    return all_samples

def analysis_database_DTI_scores(th_score):
    KEGG_DTI = pd.read_csv('KEGG_DTI_match.csv')
    ChEMBL_DTI = pd.read_csv('ChEMBL_DTI_match.csv')
    print('KEGG database: ')
    KEGG_samples = get_mean_of_other_DTI(KEGG_DTI, th_score, name='KEGG')
    print('ChEMBL database: ')
    ChEMBL_samples = get_mean_of_other_DTI(ChEMBL_DTI, th_score, name='ChEMBL')
    return KEGG_samples, ChEMBL_samples

def cal_predict_score(th_score):
    print('all matched DTIs')
    KEGG_DTI = pd.read_csv('KEGG_DTI_match.csv')
    ChEMBL_DTI = pd.read_csv('ChEMBL_DTI_match.csv')
    print('match DTIs of KEGG database: ', len(KEGG_DTI))
    print('match DTIs of ChEMBL database: ', len(ChEMBL_DTI))
    all_match_DTI = pd.concat([KEGG_DTI, ChEMBL_DTI]).drop_duplicates(keep=False).reset_index(drop=True)
    all_match_DTI['score'] = 0
    print('all match DTIs: ', len(all_match_DTI))
    for i in range(len(all_match_DTI)):
        this_drug = all_match_DTI.iloc[i, 0]
        this_protein = all_match_DTI.iloc[i, 1]
        this_score = All_scores.loc[this_drug, this_protein]
        all_match_DTI.loc[i, 'score'] = this_score
    # print(len(all_match_DTI[all_match_DTI['score'] >= th_score]))
    # get_mean_of_other_DTI(all_match_DTI, th_score)
    need_predict_DTIs1 = all_match_DTI[all_match_DTI['score'] >= th_score]
    # need_predict_DTIs1.to_csv('TP_DTI.csv', index=False)
    print(len(need_predict_DTIs1))
    mis_predict_DTIs = all_match_DTI[all_match_DTI['score'] < 0.5]
    print(len(mis_predict_DTIs))
    # need_predict_DTIs.to_csv('need_predict_DTIs.csv', index=False)

if __name__ == '__main__':
    th_score = 0.5
    get_mean_of_P_N()
    all_df = analysis_all_DTI_scores(th_score)
    KEGG_df, ChemBL_df = analysis_database_DTI_scores(th_score)
    output_df = pd.merge(all_df, KEGG_df, how='inner', on='Interval')
    output_df = pd.merge(output_df, ChemBL_df, how='inner', on='Interval')
    # output_df.to_csv('all_scores_distribution.csv', index=False)
    cal_predict_score(th_score)

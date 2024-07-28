import pandas as pd
from pathlib import Path
import os

def Make_path(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        os.makedirs(data_path)

def anti_join(data1, data2):
    data_new = pd.merge(data1, data2, indicator=True, on=['Drugbank_id', 'Uniprot_id'], how='outer').query(
        '_merge=="left_only"').drop('_merge', axis=1).reset_index(drop=True)
    return data_new


def get_all_scores():
    All_scores = pd.read_csv('../case study/All_scores_10fold_e.csv')
    Drug_id = pd.read_csv('../datasets_DTI/datasets/DTI/Drug_id.csv')
    Protein_id = pd.read_csv('../datasets_DTI/datasets/DTI/Protein_id.csv')
    All_scores.index = Drug_id['Drugbank_id'].values.tolist()
    All_scores.columns = Protein_id['Uniprot_id'].values.tolist()
    DTI_P = pd.read_csv('../datasets_DTI/datasets/DTI/DTI_P.csv')
    DTI_N = pd.read_csv('../datasets_DTI/datasets/DTI/DTI_N.csv')

    ALl_score_data = []
    for i in range(len(Drug_id)):
        this_drug = Drug_id['Drugbank_id'][i]
        for j in range(len(Protein_id)):
            this_protein = Protein_id['Uniprot_id'][j]
            this_score = All_scores.loc[this_drug, this_protein]
            ALl_score_data.append([this_drug, this_protein, this_score])
    pd_All_score = pd.DataFrame(ALl_score_data)
    pd_All_score.columns = ['Drugbank_id', 'Uniprot_id', 'Score']
    DTI_P_scores = pd.merge(pd_All_score, DTI_P, on=['Drugbank_id', 'Uniprot_id'], how='inner')
    pd_1_score = anti_join(pd_All_score, DTI_P)
    pd_2_score = anti_join(pd_1_score, DTI_N)
    return pd_2_score, DTI_P_scores



if __name__ == '__main__':
    all_scores, DTI_scores = get_all_scores()
    DrugBank_id_with_3D = pd.read_csv('DrugBank_ID_with_3D.csv')
    DrugBank_id_with_3D.columns = ['Drugbank_id']
    filter_scores = pd.merge(all_scores, DrugBank_id_with_3D, on='Drugbank_id', how='inner')
    print(len(all_scores), len(filter_scores))
    DTI_scores = pd.merge(DTI_scores, DrugBank_id_with_3D, on='Drugbank_id', how='inner')

    analysis_proteins = ['O75469', 'P35968', 'P07949', 'Q00534', 'P29317', 'Q04912']
    Make_path('dock_pairs')
    for this_protein in analysis_proteins:
        need_score = filter_scores[filter_scores['Uniprot_id'] == this_protein].sort_values(by='Score',
            ascending=False).reset_index(drop=True)
        need_top_score = need_score.head(10)
        need_end_score = need_score.tail(10)
        combine_score = pd.concat([need_top_score, need_end_score])
        combine_score.to_csv('dock_pairs/pairs_top_end_' + this_protein + '.csv', index=False)

        this_DTI_score = DTI_scores[DTI_scores['Uniprot_id'] == this_protein].reset_index(drop=True)
        this_DTI_score.to_csv('dock_pairs/pairs_DTI_' + this_protein + '.csv', index=False)


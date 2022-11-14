library(dplyr)
extra_KEGG_DTI = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/extra_DTIs/KEGG.csv",header= T)
extra_DrugCentral_DTI = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/extra_DTIs/DrugCentral.csv",header= T)
extra_CHEMBL_DTI = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/extra_DTIs/CHEMBL.csv",header= T)

drug_top10 = read.csv("D:/Users/czx/PycharmProjects/2-Experiment/case studies/drug_top10.csv",header= T)
colnames(drug_top10)[1]="Drug"
colnames(drug_top10)[2]="Protein"

Uniprot_name = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/uniprot_protein.csv",header= T)
colnames(Uniprot_name)[1]="Protein"

DrugBank_link = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/drugbank_all_drug_links/drug links.csv",header= T)
DrugBank_name = DrugBank_link[,c(1,2)]
colnames(DrugBank_name)[1]="Drug"

drug_top10_candidate = merge(drug_top10,DrugBank_name,by="Drug")
drug_top10_candidates = merge(drug_top10_candidate,Uniprot_name,by="Protein")


match_KEGG_DTI = semi_join(drug_top10_candidates,extra_KEGG_DTI,by=c("Drug","Protein"))
match_DrugCentral = semi_join(drug_top10_candidates,extra_DrugCentral_DTI,by=c("Drug","Protein"))
match_CHEMBL_DTI = semi_join(drug_top10_candidates,extra_CHEMBL_DTI,by=c("Drug","Protein"))



know_DTI = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/DTI-benchmark_set/DTI_8020.csv",header= T)
a <- rle(sort(know_DTI$X0))
m = data.frame(Drug=a$values, Number=a$lengths)

need1 = merge(match_KEGG_DTI,m,by="Drug")
need2 = merge(match_DrugCentral,m,by="Drug")
need3 = merge(match_CHEMBL_DTI,m,by="Drug")

need1s = need1[,c(1,2)]
need2s = need2[,c(1,2)]
need3s = need3[,c(1,2)]
need_all = unique(rbind(need1s,need2s,need3s))



top_100_candidate = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/top_candidate/all_top100.csv",header= T)
new_top100 = merge(top_100_candidate,DrugBank_name,by="drugbank_id")
new_top100s = merge(new_top100,Uniprot_name,by="uniprot_id")


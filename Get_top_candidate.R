library(dplyr)

Predict_scores = read.csv("D:/Users/czx/PycharmProjects/2-Experiment/Predict_scores.csv",header= T)
Drugs = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/DTI-benchmark_set/Drug_1409.csv",header= T)
Proteins = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/DTI-benchmark_set/Protein_1648.csv",header= T)

drug_top_10_candi = matrix(nc=3,nr=0)
drug_top_10_candi = data.frame(drug_top_10_candi)
colnames(drug_top_10_candi)[1]="drugbank_id"
colnames(drug_top_10_candi)[2]="uniprot_id"
colnames(drug_top_10_candi)[3]="scores"

for(i in 1:length(Drugs[,1]))
{
  this_drug = data.frame(drugbank_id=Drugs[i,1])
  this_scores = semi_join(Predict_scores,this_drug,by="drugbank_id")
  rank_scores = this_scores[order(this_scores$scores,decreasing = T),]
  top_scores = rank_scores[c(1:10),]
  drug_top_10_candi = rbind(drug_top_10_candi,top_scores)
}

write.csv(drug_top_10_candi,"D:/Users/czx/PycharmProjects/2-Experiment/case studies/drug_top10.csv",row.names = FALSE)


drug_top_5_candi = matrix(nc=3,nr=0)
drug_top_5_candi = data.frame(drug_top_5_candi)
colnames(drug_top_5_candi)[1]="drugbank_id"
colnames(drug_top_5_candi)[2]="uniprot_id"
colnames(drug_top_5_candi)[3]="scores"

for(i in 1:length(Drugs[,1]))
{
  this_drug = data.frame(drugbank_id=Drugs[i,1])
  this_scores = semi_join(Predict_scores,this_drug,by="drugbank_id")
  rank_scores = this_scores[order(this_scores$scores,decreasing = T),]
  top_scores = rank_scores[c(1:5),]
  drug_top_5_candi = rbind(drug_top_5_candi,top_scores)
}

write.csv(drug_top_5_candi,"D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/top_candidate/drug_top5.csv",row.names = FALSE)


all_top_100 = Predict_scores[c(1:100),]
write.csv(all_top_100,"D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/top_candidate/all_top100.csv",row.names = FALSE)

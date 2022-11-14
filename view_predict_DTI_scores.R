library(dplyr)
extra_KEGG_DTI = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/extra_DTIs/KEGG.csv",header= T)
extra_DrugCentral_DTI = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/extra_DTIs/DrugCentral.csv",header= T)
extra_CHEMBL_DTI = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/extra_DTIs/CHEMBL.csv",header= T)

know_DTI = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/DTI-benchmark_set/DTI_8020.csv",header= T)
colnames(know_DTI)[1] = "Drug"
colnames(know_DTI)[2] = "Protein"

Predict_scores = read.csv("D:/Users/czx/PycharmProjects/2-Experiment/Predict_scores.csv",header= T)

# test = subset(Predict_scores,Predict_scores$drugbank_id=="DB00504")

colnames(Predict_scores)[1] = "Drug"
colnames(Predict_scores)[2] = "Protein"

extra_KEGG_DTI_scores = semi_join(Predict_scores,extra_KEGG_DTI,by=c("Drug","Protein"))
extra_DrugCentral_DTI_scores = semi_join(Predict_scores,extra_DrugCentral_DTI,by=c("Drug","Protein"))
extra_CHEMBL_DTI_scores = semi_join(Predict_scores,extra_CHEMBL_DTI,by=c("Drug","Protein"))


mean1 = mean(extra_KEGG_DTI_scores$scores)
mean2 = mean(extra_DrugCentral_DTI_scores$scores)
mean3 = mean(extra_CHEMBL_DTI_scores$scores)

m = as.numeric(extra_KEGG_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)


m = as.numeric(extra_DrugCentral_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)


m = as.numeric(extra_CHEMBL_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)


extra_DTIs = unique(rbind(extra_KEGG_DTI,extra_DrugCentral_DTI,extra_CHEMBL_DTI))
extra_DTIs_scores = semi_join(Predict_scores,extra_DTIs,by=c("Drug","Protein"))
m = as.numeric(extra_DTIs_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)

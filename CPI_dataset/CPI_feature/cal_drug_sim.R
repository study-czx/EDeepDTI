library(philentropy)

ECFP2 = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug/ECFP2_1024.csv",header=FALSE)
ECFP4 = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug/ECFP4_1024.csv",header=FALSE)
ECFP6 = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug/ECFP6_1024.csv",header=FALSE)

MACCS = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug/MACCS.csv",header=FALSE)
Pubchem = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug/Pubchem.csv",header=FALSE)
RDK = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug/RDK_2048.csv",header=FALSE)

# cal drug sim

ECFP2_sim = 1-distance(ECFP2, method="jaccard")
ECFP4_sim = 1-distance(ECFP4, method="jaccard")
ECFP6_sim = 1-distance(ECFP6, method="jaccard")
MACCS_sim = 1-distance(MACCS, method="jaccard")
Pubchem_sim = 1-distance(Pubchem, method="jaccard")
RDK_sim = 1-distance(RDK, method="jaccard")

write.csv(ECFP2_sim,"D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug_sim/ECFP2.csv",row.names = F)
write.csv(ECFP4_sim,"D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug_sim/ECFP4.csv",row.names = F)
write.csv(ECFP6_sim,"D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug_sim/ECFP6.csv",row.names = F)
write.csv(MACCS_sim,"D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug_sim/MACCS.csv",row.names = F)
write.csv(Pubchem_sim,"D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug_sim/Pubchem.csv",row.names = F)
write.csv(RDK_sim,"D:/Users/czx/PycharmProjects/DTI_data_Get/CPI_data/CPI_feature/drug_sim/RDK.csv",row.names = F)

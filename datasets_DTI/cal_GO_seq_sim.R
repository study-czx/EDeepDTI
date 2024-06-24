library(GOSemSim)
library(dplyr)

getwd()

get_similarity_data = function(Uniprot_GO, protein_id, GOsim_data){
  colnames(Uniprot_GO)[1] = 'Uniprot_id'
  n_protein = length(protein_id[,1])
  protein_sim = matrix(nr=n_protein,nc=n_protein)
  for(i in 1:(n_protein-1)){
    if (i %% 100 == 0){
      print(i)
    }
    for(j in (i+1):n_protein){
      protein1 = data.frame(Uniprot_id = protein_id[i,1])
      protein2 = data.frame(Uniprot_id = protein_id[j,1])
      GO_list1 = semi_join(Uniprot_GO,protein1,by="Uniprot_id")
      GO_list2 = semi_join(Uniprot_GO,protein2,by="Uniprot_id")
      GO_set1 = unlist(GO_list1[,2])
      GO_set2 = unlist(GO_list2[,2])
      protein_sim[i,j] = mgoSim(GO_set1, GO_set2, semData = GOsim_data, measure="Wang",combine = "BMA")
      protein_sim[j,i] = protein_sim[i,j]
    }
  }
  for(i in 1:n_protein){
    protein_sim[i,i] = 1
  }
  protein_sim[is.na(protein_sim)] <- 0
  return(protein_sim)
}

# DTI and CPI protein sim
cal_all_protein_GO_sim = function(){
  DTI_protein_sim_MF = get_similarity_data(All_uniprot_MF, DTI_protein_id, MF_data)
  write.csv(DTI_protein_sim_MF,file="datasets/DTI/protein_sim/MF.csv",row.names =FALSE)
  print('DTI BP')
  DTI_protein_sim_BP = get_similarity_data(All_uniprot_BP, DTI_protein_id, BP_data)
  write.csv(DTI_protein_sim_BP,file="datasets/DTI/protein_sim/BP.csv",row.names =FALSE)
  print('DTI CC')
  DTI_protein_sim_CC = get_similarity_data(All_uniprot_CC, DTI_protein_id, CC_data)
  write.csv(DTI_protein_sim_CC,file="datasets/DTI/protein_sim/CC.csv",row.names =FALSE)
  print('CPI MF')
  CPI_protein_sim_MF = get_similarity_data(All_uniprot_MF, CPI_protein_id, MF_data)
  write.csv(CPI_protein_sim_MF,file="datasets/CPI/protein_sim/MF.csv",row.names =FALSE)
  print('CPI BP')
  CPI_protein_sim_BP = get_similarity_data(All_uniprot_BP, CPI_protein_id, BP_data)
  write.csv(CPI_protein_sim_BP,file="datasets/CPI/protein_sim/BP.csv",row.names =FALSE)
  print('CPI CC')
  CPI_protein_sim_CC = get_similarity_data(All_uniprot_CC, CPI_protein_id, CC_data)
  write.csv(CPI_protein_sim_CC,file="datasets/CPI/protein_sim/CC.csv",row.names =FALSE)
}

# DTI and CPI protein sim
cal_GO_sim_DK = function(){
  print('Davis MF')
  Davis_sim_MF = get_similarity_data(Davis_uniprot_MF, Davis_protein_id, MF_data)
  write.csv(Davis_sim_MF,file="datasets/Davis_5fold/protein_sim/MF.csv",row.names =FALSE)
  print('Davis BP')
  Davis_sim_BP = get_similarity_data(Davis_uniprot_BP, Davis_protein_id, BP_data)
  write.csv(Davis_sim_BP,file="datasets/Davis_5fold/protein_sim/BP.csv",row.names =FALSE)
  print('Davis CC')
  Davis_sim_CC = get_similarity_data(Davis_uniprot_CC, Davis_protein_id, CC_data)
  write.csv(Davis_sim_CC,file="datasets/Davis_5fold/protein_sim/CC.csv",row.names =FALSE)
  print('KIBA MF')
  KIBA_sim_MF = get_similarity_data(KIBA_uniprot_MF, KIBA_protein_id, MF_data)
  write.csv(KIBA_sim_MF,file="datasets/KIBA_5fold/protein_sim/MF.csv",row.names =FALSE)
  print('KIBA BP')
  KIBA_sim_BP = get_similarity_data(KIBA_uniprot_BP, KIBA_protein_id, BP_data)
  write.csv(KIBA_sim_BP,file="datasets/KIBA_5fold/protein_sim/BP.csv",row.names =FALSE)
  print('KIBA CC')
  KIBA_sim_CC = get_similarity_data(KIBA_uniprot_CC, KIBA_protein_id, CC_data)
  write.csv(KIBA_sim_CC,file="datasets/KIBA_5fold/protein_sim/CC.csv",row.names =FALSE)
}

library(Rcpi)
cal_protein_seq_sim = function(){
  DTI_seq = readFASTA("datasets/DTI/protein.fasta")
  DTI_seq_Sim = round(calcParProtSeqSim(DTI_seq, cores = 12, type = "local", submat = "BLOSUM62"),6)
  id = unlist(DTI_protein_id)
  colnames(DTI_seq_Sim)=id
  rownames(DTI_seq_Sim)=id
  write.csv(DTI_seq_Sim,file="datasets/DTI/protein_sim/seq.csv",row.names =FALSE)
  
  CPI_seq = readFASTA("datasets/CPI/all_protein.fasta")
  CPI_seq_Sim = round(calcParProtSeqSim(CPI_seq, cores = 12, type = "local", submat = "BLOSUM62"),6)
  id = unlist(CPI_protein_id)
  colnames(CPI_seq_Sim)=id
  rownames(CPI_seq_Sim)=id
  write.csv(CPI_seq_Sim,file="datasets/CPI/protein_sim/seq.csv",row.names =FALSE)
} 

cal_DK_seq_sim = function(){
  Davis_seq = readFASTA("datasets/Davis_5fold/protein.fasta")
  Davis_seq_Sim = round(calcParProtSeqSim(Davis_seq, cores = 12, type = "local", submat = "BLOSUM62"),6)
  id = unlist(Davis_protein_id)
  colnames(Davis_seq_Sim)=id
  rownames(Davis_seq_Sim)=id
  write.csv(Davis_seq_Sim,file="datasets/Davis_5fold/protein_sim/seq.csv",row.names =FALSE)
  
  KIBA_seq = readFASTA("datasets/KIBA_5fold/protein.fasta")
  KIBA_seq_Sim = round(calcParProtSeqSim(KIBA_seq, cores = 12, type = "local", submat = "BLOSUM62"),6)
  id = unlist(KIBA_protein_id[,1])
  colnames(KIBA_seq_Sim)=id
  rownames(KIBA_seq_Sim)=id
  write.csv(KIBA_seq_Sim,file="datasets/KIBA_5fold/protein_sim/seq.csv",row.names =FALSE)
} 

MF_data <- godata('org.Hs.eg.db', ont="MF", computeIC=FALSE)
BP_data <- godata('org.Hs.eg.db', ont="BP", computeIC=FALSE)
CC_data <- godata('org.Hs.eg.db', ont="CC", computeIC=FALSE)

# DTI and CPI dataset GO sim
All_uniprot_MF = read.csv("processed_data/Uniprot_MF.csv",header=T,encoding = "UTF-8")
All_uniprot_BP = read.csv("processed_data/Uniprot_BP.csv",header=T,encoding = "UTF-8")
All_uniprot_CC = read.csv("processed_data/Uniprot_CC.csv",header=T,encoding = "UTF-8")

DTI_protein_id = read.csv("datasets/DTI/Protein_id.csv",header=T)
CPI_protein_id = read.csv("datasets/CPI/all_protein_id.csv",header=T)

cal_all_protein_GO_sim()

# DTI and CPI seq sim
cal_protein_seq_sim()

###########################################################################################################################
# Davis and KIBA
Davis_protein_id = read.csv("datasets/Davis_5fold/Uniprot_id.csv",header=T)
KIBA_protein_id = read.csv("datasets/KIBA_5fold/Protein.csv",header=T)

cal_DK_seq_sim()

Davis_uniprot_MF = read.csv("datasets/Davis_5fold/GO/Uniprot_MF.csv",header=T,encoding = "UTF-8")
Davis_uniprot_BP = read.csv("datasets/Davis_5fold/GO/Uniprot_BP.csv",header=T,encoding = "UTF-8")
Davis_uniprot_CC = read.csv("datasets/Davis_5fold/GO/Uniprot_CC.csv",header=T,encoding = "UTF-8")

KIBA_uniprot_MF = read.csv("datasets/KIBA_5fold/GO/Uniprot_MF.csv",header=T,encoding = "UTF-8")
KIBA_uniprot_BP = read.csv("datasets/KIBA_5fold/GO/Uniprot_BP.csv",header=T,encoding = "UTF-8")
KIBA_uniprot_CC = read.csv("datasets/KIBA_5fold/GO/Uniprot_CC.csv",header=T,encoding = "UTF-8")

cal_GO_sim_DK()
















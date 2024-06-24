getwd()

library(rcdk)
library(philentropy)
library(fingerprint)

# DTI dataset
drug_smiles = read.csv("datasets/DTI/drug_smiles.csv",header=T)
id = drug_smiles[,1]
smiles = drug_smiles[,2]
sp <- get.smiles.parser()
mols <- parse.smiles(smiles)
fps = lapply(mols,get.fingerprint,type='pubchem')
f = fp.to.matrix(fps)
write.csv(f,file="datasets/DTI/drug_finger/PubChem.csv",row.names =FALSE)

# CPI dataset
compound_smiles = read.csv("datasets/CPI/all_compound_smiles.csv",header=T)
id = compound_smiles[,1]
smiles = compound_smiles[,2]

result_data <- data.frame()
chunk_size = 100
round_times <- length(id) %/% chunk_size
remain_round <- length(id) %% chunk_size
sp <- get.smiles.parser()

# Prevent OutOfMemoryError by computing in batches
for (i in 1:(round_times+1)) {
  print(i)
  start_index <- (i - 1) * chunk_size + 1
  end_index <- min(i * chunk_size, length(id))
  current_smiles <- smiles[start_index:end_index]
  mols <- parse.smiles(current_smiles)
  fps = lapply(mols,get.fingerprint,type='pubchem')
  f = fp.to.matrix(fps)
  result_data = rbind(result_data, f)
  gc()
}
write.csv(result_data, file="datasets/CPI/drug_finger/PubChem.csv",row.names =FALSE)


# Davis dataset and KIBA dataset
drug_smiles = read.csv("datasets/Davis_5fold/Drug.csv",header=T)
id = drug_smiles[,1]
smiles = drug_smiles[,2]
sp <- get.smiles.parser()
mols <- parse.smiles(smiles)
fps = lapply(mols,get.fingerprint,type='pubchem')
f = fp.to.matrix(fps)
write.csv(f,file="datasets/Davis_5fold/drug_finger/PubChem.csv",row.names =FALSE)

# KIBA
drug_smiles = read.csv("datasets/KIBA_5fold/Drug.csv",header=T)
id = drug_smiles[,1]
smiles = drug_smiles[,2]
sp <- get.smiles.parser()
mols <- parse.smiles(smiles)
fps = lapply(mols,get.fingerprint,type='pubchem')
f = fp.to.matrix(fps)
write.csv(f,file="datasets/KIBA_5fold/drug_finger/PubChem.csv",row.names =FALSE)

############################################################################################################

# get protein fasta to calculate Descriptors
library(seqinr)

# DTI dataset
protein_seq = read.csv("datasets/DTI/protein_sequence.csv",header=T)
id = protein_seq[,1]
seq = protein_seq[,2]

header_name = matrix(nr=length(id),nc=1)
for(i in 1:length(id))
{
  id1 = id[i]
  header = paste(id1,1,sep = "|")
  header = paste(header,"training",sep = "|")
  header_name[i] = header
}
header_name = as.list(header_name)
sequence = as.list(seq)
write.fasta(sequence, names = header_name, file='datasets/DTI/protein.fasta', open = "w", nbchar = 60, as.string = FALSE)


# CPI dataset
protein_seq = read.csv("datasets/CPI/all_protein_sequence.csv",header=T)
id = protein_seq[,1]
seq = protein_seq[,2]

header_name = matrix(nr=length(id),nc=1)
for(i in 1:length(id))
{
  id1 = id[i]
  header = paste(id1,1,sep = "|")
  header = paste(header,"training",sep = "|")
  header_name[i] = header
}
header_name = as.list(header_name)
sequence = as.list(seq)
write.fasta(sequence, names = header_name, file='datasets/CPI/all_protein.fasta', open = "w", nbchar = 60, as.string = FALSE)


# Davis dataset and KIBA dataset
protein_seq = read.csv("datasets/Davis_5fold/Protein.csv",header=T)
id = protein_seq[,1]
seq = protein_seq[,2]

header_name = matrix(nr=length(id),nc=1)
for(i in 1:length(id))
{
  id1 = id[i]
  header = paste(id1,1,sep = "|")
  header = paste(header,"training",sep = "|")
  header_name[i] = header
}
header_name = as.list(header_name)
sequence = as.list(seq)
write.fasta(sequence, names = header_name, file='datasets/Davis_5fold/protein.fasta', open = "w", nbchar = 60, as.string = FALSE)

# KIBA
protein_seq = read.csv("datasets/KIBA_5fold/Protein.csv",header=T)
id = protein_seq[,1]
seq = protein_seq[,2]

header_name = matrix(nr=length(id),nc=1)
for(i in 1:length(id))
{
  id1 = id[i]
  header = paste(id1,1,sep = "|")
  header = paste(header,"training",sep = "|")
  header_name[i] = header
}
header_name = as.list(header_name)
sequence = as.list(seq)
write.fasta(sequence, names = header_name, file='datasets/KIBA_5fold/protein.fasta', open = "w", nbchar = 60, as.string = FALSE)

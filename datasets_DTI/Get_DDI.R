getwd()

library(XML)
library(dplyr)
drugbank = xmlParse(file = "origin_data/DrugBank/full database.xml")
rootnode = xmlRoot(drugbank)
rootsize <- xmlSize(rootnode)


print(rootsize)
db = matrix(nrow = rootsize,ncol = 4)
for (i in 1:rootsize)
{
  db[i,1] = xmlGetAttr(rootnode[[i]],"type")
  db[i,2] = xmlValue(rootnode[[i]][["drugbank-id",primary = TRUE]])
  db[i,3] = xmlValue(rootnode[[i]][["groups"]][[1]])
  if(xmlSize(rootnode[[i]][["drug-interactions"]])!=0)
  {
    for(j in 1:xmlSize(rootnode[[i]][["drug-interactions"]]))
    {
      interid = xmlValue(rootnode[[i]][["drug-interactions"]][[j]][["drugbank-id"]])
      if(is.na(db[i,4])==TRUE)
      {
        db[i,4] = interid
      }
      else
      {
        db[i,4] = paste(db[i,4],interid,sep = ",")
      }
    }
  }
}
db_new = data.frame(db)

# db_slected1=subset(db_new,db_new[,3]=="approved")          
db_slected = subset(db_new, db_new[,1]=="small molecule")
db_slected = db_slected[,c(2,4)]
db_slected = na.omit(db_slected)

Drugbank_id = db_slected[,1]
db_need = db_slected

db_network = matrix(nrow = 10000000,ncol = 2)
k=1
for(i in 1:length(Drugbank_id))
{
  x=strsplit(db_need[i,2],",")
  y=as.array(unlist(x))
  for(j in 1:length(y))
  {
    db_network[k,1]=db_need[i,1]
    db_network[k,2]=y[j]
    k = k+1
  }
}
db_n = data.frame(na.omit(db_network))
colnames(db_n)[1] = 'drugbank_id1'
colnames(db_n)[2] = 'drugbank_id2'
write.csv(db_n,file="Drugbank_DDI_2560048.csv",row.names =FALSE,quote = F)


setwd("~/Projects/carmapy_light/tests")

library(data.table)
library(CARMA)
library(RcppCNPy)

#test1

suff="test1"
z_full=fread("01_test_z.csv",header=T,sep=",",data.table=F)
ld=fread("01_test_ld.csv",header=F,sep=",",data.table=F)
z=z_full[,"z"]
R3=CARMA(z.list = list(z),ld.list = list(ld),lambda.list = list(1),
         output.labels = NULL,all.iter = 1)
R3=R3[[1]]
pip=cbind(PIP=R3$PIPs)

fwrite(paste0(suff,"_o.csv"),x=R3$Outliers,col.names = T,row.names=F)
fwrite(paste0(suff,"_pip.csv"),x=pip,col.names = T,row.names=F)

#test2

suff="test2"
z_full=fread("02_test_z.csv",header=T,sep=",",data.table=F)
ld=npyLoad("02_test_ld.npy")
z=z_full[,"z"]
R3=CARMA(z.list = list(z),ld.list = list(ld),lambda.list = list(1),
         output.labels = NULL,all.iter = 1)
R3=R3[[1]]
pip=cbind(PIP=R3$PIPs)

fwrite(paste0(suff,"_o.csv"),x=R3$Outliers,col.names = T,row.names=F)
fwrite(paste0(suff,"_pip.csv"),x=pip,col.names = T,row.names=F)

#test3

suff="test3"
z_full=fread("03_test_z.csv",header=T,sep=",",data.table=F)
ld=fread("03_test_ld.csv",header=F,sep=",",data.table=F)
z=z_full[,"z"]
R3=CARMA(z.list = list(z),ld.list = list(ld),lambda.list = list(1),
         output.labels = NULL,all.iter = 1)
R3=R3[[1]]
pip=cbind(PIP=R3$PIPs)

fwrite(paste0(suff,"_o.csv"),x=R3$Outliers,col.names = T,row.names=F)
fwrite(paste0(suff,"_pip.csv"),x=pip,col.names = T,row.names=F)


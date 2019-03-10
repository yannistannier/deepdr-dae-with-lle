library(FactoMineR)
library(dimRed)
library(reshape2)
library(ggplot2)
library(FactoMineR)
library(tm)
library(stringr)
library(NMIcode)
library(LICORS)
library(readr)
library(keras)
library(mclust)
pen.tra = read.table("/Users/jzk/Documents/M2/reducDimold/penDigitss/pendigits.tra", sep = ",")
pen.tes = read.table("/Users/jzk/Documents/M2/reducDimold/penDigitss/pendigits.tes", sep = ",")
pen = rbind(pen.tra, pen.tes)



dim(pen.tra)
X.train=pen.tra[,-17]
Class.train=pen.tra[,17]
X.test=pen.tes[,-17]
Class.test=pen.tes[,17]


library(tensorflow)
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = FALSE)
dim(mnist$train$images)
X.train=mnist$test$images
Class.train=as.vector(mnist$test$labels)
X.train=mnist$train$images
Class.train=as.vector(mnist$train$labels)



source("http://bioconductor.org/biocLite.R")
biocLite("rhdf5")
library(rhdf5)
usps=h5read("/Users/jzk/Documents/M2/reducDim/usps.h5","/train")
usps_tr=usps$data
usps_class=usps$target
X.train=t(usps_tr)
Class.train=as.vector(usps_class)


X.train=read_csv("/Users/jzk/Documents/M2/reducDimold/fashionmnist/fashion-mnist_test.csv",col_types = cols(.default = "i"))
Class.train=X.train$label
head(X.train)
X.train=X.train[,-1]
X.train=as.matrix(X.train)
dim(X.train)




#########ACP
library(FactoMineR)
library(aricode)
library(MLmetrics)
library(caret)
PCA=FactoMineR::PCA(X.train)
barplot(PCA$eig[,1],main="Eigenvalues",names.arg=1:nrow(PCA$eig))
summary(PCA)
PCA$ind$coord
NMI=c()
ARI=c()
clustering=Mclust(PCA$ind$coord,G=10)
for(i in 1:10){
  clustering=Mclust(PCA$ind$coord,G=10)
  NMI=cbind(NMI,NMI(clustering$classification,Class.train))
  ARI=cbind(ARI,ARI(clustering$classification,Class.train))
}
NMIPCA=as.vector(NMI);mean(NMIPCA)
ARIPCA=as.vector(ARI);mean(ARIPCA)
boxplot(NMIPCA)
boxplot(ARIPCA)
write.csv(NMIPCA,"/Users/jzk/Documents/M2/projet/NMI/NMIPCAFMNIST.csv")
write.csv(ARIPCA,"/Users/jzk/Documents/M2/projet/ARI/ARIPCAFMNIST.csv")


####KPCA
library(kernlab)
KPCA=emb2 <- embed(X.train, "kPCA")
KPCA <- kpca(X.train)
KPCA <- kpca(~.,data=X.train,kernel="rbfdot",kpar=list(sigma=0.2),features=2)
slot(KPCA,"xmatrix")
NMI=c()
ARI=c()
for(i in 1:10){
  clustering=kmeans(slot(KPCA,"xmatrix")[,c(1:2)],10)
  NMI=cbind(NMI,NMI(clustering$cluster,Class.train))
  ARI=cbind(ARI,ARI(clustering$cluster,Class.train))
}
NMIKPCA=as.vector(NMI)
ARIKPCA=as.vector(ARI)
boxplot(NMIKPCA)
boxplot(ARIKPCA)




####Isomap
library(dimRed)
ISO <- embed(X.train, "Isomap", .mute = NULL, knn = 15,ndim=5)
plot(ISO, type = "2vars")
red=ISO@data@data
NMI=c()
ARI=c()
clustering=Mclust(red,G=10)
for(i in 1:10){
  clustering=Mclust(red,G=10)
  NMI=cbind(NMI,NMI(clustering$classification,Class.train))
  ARI=cbind(ARI,ARI(clustering$classification,Class.train))
}
NMIISO=as.vector(NMI)
ARIISO=as.vector(ARI)
boxplot(NMIISO)
boxplot(ARIISO)
write.csv(NMIISO,"/Users/jzk/Documents/M2/projet/NMI/NMIISOFMNIST.csv")
write.csv(ARIISO,"/Users/jzk/Documents/M2/projet/ARI/ARIISOFMNIST.csv")


##MDS
library(MASS)
d <- dist(X.train,method="euclidean") # euclidean distances between the rows
fit <- isoMDS(d, k=7) # k is the number of dim
fit # view results
red=fit$points
red=read_csv("/Users/jzk/Documents/M2/projet/projet/MDS.csv");red=as.matrix(red)
Class.train=read_csv("/Users/jzk/Documents/M2/projet/projet/label.csv")
Class.train=Class.train$`7.000000000000000000e+00`
NMI=c()
ARI=c()
clustering=Mclust(red,G=10)
for(i in 1:10){
  clustering=Mclust(red,G=10)
  NMI=cbind(NMI,NMI(clustering$classification,Class.train))
  ARI=cbind(ARI,ARI(clustering$classification,Class.train))
}
NMIMDS=as.vector(NMI)
ARIMDS=as.vector(ARI)
boxplot(NMIMDS)
boxplot(ARIMDS)
write.csv(NMIMDS,"/Users/jzk/Documents/M2/projet/NMI/NMIMDSFMNIST.csv")
write.csv(ARIMDS,"/Users/jzk/Documents/M2/projet/ARI/ARIMDSFMNIST.csv")

NMIMDS
###LLE
library(lle)
red <- lle(X.train, m=2, k=10, reg=2, ss=FALSE, id=TRUE, v=0.9 )
red=read_csv("/Users/jzk/Documents/M2/projet/projet/LLE.csv");red=as.matrix(red)
red=red$Y
NMI=c()
ARI=c()
clustering=Mclust(red,G=10)
for(i in 1:10){
  clustering=Mclust(red,G=10)
  NMI=cbind(NMI,NMI(clustering$classification,Class.train))
  ARI=cbind(ARI,ARI(clustering$classification,Class.train))
}
NMILLE=as.vector(NMI)
ARILLE=as.vector(ARI)
boxplot(NMILLE)
boxplot(ARILLE)
write.csv(NMILLE,"/Users/jzk/Documents/M2/projet/NMI/NMILLEFMNIST.csv")
write.csv(ARILLE,"/Users/jzk/Documents/M2/projet/ARI/ARILLEFMNIST.csv")

library(Rdimtools)
red=do.ltsa(X.train, ndim = 5, type =c("proportion",0.1))
red=red$Y
red=read_csv("/Users/jzk/Documents/M2/projet/projet/LTSA.csv")
Class.train=read_csv("/Users/jzk/Documents/M2/projet/projet/label.csv")
Class.train=Class.train$`7.000000000000000000e+00`
red=as.matrix(red)
NMI=c()
ARI=c()
clustering=Mclust(red,G=10)
for(i in 1:10){
  clustering=Mclust(red,G=10)
  NMI=cbind(NMI,NMI(clustering$classification,Class.train))
  ARI=cbind(ARI,ARI(clustering$classification,Class.train))
}
NMILLE=as.vector(NMI)
ARILLE=as.vector(ARI)
boxplot(NMILLE);mean(NMILLE)
boxplot(ARILLE);mean(ARILLE)
write.csv(NMILLE,"/Users/jzk/Documents/M2/projet/NMI/NMILTSAFMNIST.csv")
write.csv(ARILLE,"/Users/jzk/Documents/M2/projet/ARI/ARILTSAFMNIST.csv")

###UMAP

red=read_csv("/Users/jzk/Documents/M2/projet/projet/UMAP.csv")
Class.train=read_csv("/Users/jzk/Documents/M2/projet/projet/label.csv")
Class.train=Class.train$`7.000000000000000000e+00`
red=as.matrix(red)
NMI=c()
ARI=c()
clustering=Mclust(red,G=10)
for(i in 1:10){
  clustering=Mclust(red,G=10)
  NMI=cbind(NMI,NMI(clustering$classification,Class.train))
  ARI=cbind(ARI,ARI(clustering$classification,Class.train))
}
NMILLE=as.vector(NMI)
ARILLE=as.vector(ARI)
boxplot(NMILLE);mean(NMILLE)
boxplot(ARILLE);mean(ARILLE)
write.csv(NMILLE,"/Users/jzk/Documents/M2/projet/NMI/NMIUMAPFMNIST.csv")
write.csv(ARILLE,"/Users/jzk/Documents/M2/projet/ARI/ARIUMAPFMNIST.csv")


library(Matrix)
library(NMF)
library(readr)
library(Matrix)
library(NMF)
library(tidytext)
library(tm)
library(slam)
library(dplyr)
library(SnowballC)
library(skmeans)
library(textir)
library(stm)
library(factoextra)
library(foreach)
library(doParallel)
library(fastICA)
library(wordcloud)
library(topicmodels)
data_used.tfidf=X.train
weight=Matrix(rep(1,dim(data_used.tfidf)[1]*dim(data_used.tfidf)[2]),nrow=dim(data_used.tfidf)[1]);dim(weight)
res=nmf(X.train,10,method="ls-nmf", .options="vt",seed='nndsvd',weight=as.matrix(weight))
res.coef <- coef(res)####on r??cup??re H
res.bas <- basis(res)####on r??cup??re W
heatmap(res.bas)
red=res.bas

NMI=c()
ARI=c()
clustering=Mclust(red,G=10)
for(i in 1:10){
  clustering=Mclust(red,G=10)
  NMI=cbind(NMI,NMI(clustering$classification,Class.train))
  ARI=cbind(ARI,ARI(clustering$classification,Class.train))
}
NMILLE=as.vector(NMI)
ARILLE=as.vector(ARI)
boxplot(NMILLE)
boxplot(ARILLE)
write.csv(NMILLE,"/Users/jzk/Documents/M2/projet/NMI/NMINMFMNIST.csv")
write.csv(ARILLE,"/Users/jzk/Documents/M2/projet/ARI/ARINMFMNIST.csv")

####On load les donn??es
NMIAE <- read_csv("~/Documents/M2/projet/NMI/AE_NMI-2.csv")
NMIAE=NMIAE$x

NMIAE=c(0.619148480401683,0.622058475409507,0.599357598184059,0.611669946777276,0.612980116108482,0.6194605223113,0.611809445804786,
        0.609646195077058,
        0.620292913556716,
        0.629093044390368)

NMINMF=read_csv("~/Documents/M2/projet/NMI/NMF_NMI.csv")
NMINMF=NMINMF$x

NMIAELLE=read_csv("/Users/jzk/Downloads/DAELEE2_NMI.csv")
NMIAELLE=NMIAELLE$x

NMIAELLE=c(0.613604966649526,
           0.625035309545582,
           0.641735829193394,
           0.638362630665628,
           0.633768885678127,
           0.64301349189263,
           0.633658811165764,
           0.609628723016092,
           0.636312615366251,
           0.613706282593635)

NMINMF=read_csv("/Users/jzk/Documents/M2/projet/NMI/NMINMFMNIST.csv")
NMINMF=NMINMF$x

NMIPCA=read_csv("/Users/jzk/Documents/M2/projet/NMI/NMIPCAFMNIST.csv")
NMIPCA=NMIPCA$x

NMIKPCA=read_csv("/Users/jzk/Documents/M2/projet/NMI/KERNALPCA_NMI.csv")
NMIKPCA=NMIKPCA$x

NMIMDS=read_csv("/Users/jzk/Documents/M2/projet/NMI/NMIMDSFMNIST.csv")
NMIMDS=NMIMDS$x

NMILTSA=read_csv("/Users/jzk/Documents/M2/projet/NMI/NMILTSAFMNIST.csv")
NMILTSA=NMILTSA$x

NMIISO=read_csv("/Users/jzk/Documents/M2/projet/NMI/NMIISOFMNIST.csv")
NMIISO=NMIISO$x

NMILLE=read_csv("/Users/jzk/Documents/M2/projet/NMI/NMILLEFMNIST.csv")
NMILLE=NMILLE$x

NMIUMAP=read_csv("/Users/jzk/Documents/M2/projet/NMI/NMIUMAPFMNIST.csv")
NMIUMAP=NMIUMAP$x

boxplot(NMIPCA,NMINMF,NMIMDS,NMIISO,NMILLE,NMILTSA,NMIAE,NMIUMAP,NMIAELLE,names=c("PCA","NMF","MDS","ISOMAP","LLE","LTSA","AE","UMAP","DeepDr"))

####On load les donn??es  ARI
ARIAE <- read_csv("~/Documents/M2/projet/NMI/AE_ARI-2.csv")
ARIAE=ARIAE$x
ARIAE=c(0.471868884412653,0.473620546320627,0.454919443867452,0.459938250291911,0.461576024312166,0.45192332803295,
0.460523771935872,
0.48255091040172,
0.471828078063006,
0.500065401234325)

ARIAELLE=read_csv("/Users/jzk/Downloads/DAELLE2_ARI.csv")
ARIAELLE=ARIAELLE$x
ARIAELLE=c(0.464421652293046,
           0.482780377909208,
           0.519405790093007,
           0.51715003262984,
           0.491345880509513,
           0.515244485708418,
           0.507963014939459,
           0.466197618532168,
           0.51884289802953,
           0.464945734338518)

ARINMF=read_csv("/Users/jzk/Documents/M2/projet/ARI/ARINMFMNIST.csv")
ARINMF=ARINMF$x

ARIPCA=read_csv("/Users/jzk/Documents/M2/projet/ARI/ARIPCAFMNIST.csv")
ARIPCA=ARIPCA$x

ARIMDS=read_csv("/Users/jzk/Documents/M2/projet/ARI/ARIMDSFMNIST.csv")
ARIMDS=ARIMDS$x


ARILTSA=read_csv("/Users/jzk/Documents/M2/projet/ARI/ARILTSAFMNIST.csv")
ARILTSA=ARILTSA$x

ARIISO=read_csv("/Users/jzk/Documents/M2/projet/ARI/ARIISOFMNIST.csv")
ARIISO=ARIISO$x

ARILLE=read_csv("/Users/jzk/Documents/M2/projet/ARI/ARILLEFMNIST.csv")
ARILLE=ARILLE$x


ARINMF=read_csv("/Users/jzk/Documents/M2/projet/ARI/ARINMFMNIST.csv")
ARINMF=ARINMF$x

ARIUMAP=read_csv("/Users/jzk/Documents/M2/projet/ARI/ARIUMAPFMNIST.csv")
ARIUMAP=ARIUMAP$x

boxplot(ARIPCA,ARINMF,ARIMDS,ARIISO,ARILLE,ARILTSA,ARIAE,ARIUMAP,ARIAELLE,names=c("PCA","NMF","MDS","ISOMAP","LLE","LTSA","AE","UMAP","DeepDr"))


t.test(NMIAELLE,NMIPCA, paired = TRUE, alternative = "greater")
t.test(NMIAELLE, NMINMF, paired = TRUE, alternative = "greater")
t.test(NMIAELLE, NMIMDS, paired = TRUE, alternative = "greater")
t.test(NMIAELLE, NMIISO, paired = TRUE, alternative = "greater")
t.test(NMIAELLE, NMILLE, paired = TRUE, alternative = "greater")
t.test(NMIAELLE, NMILTSA, paired = TRUE, alternative = "greater")
t.test(NMIAELLE, NMIAE, paired = TRUE, alternative = "greater")
t.test(NMIAELLE, NMIUMAP, paired  = FALSE ,alternative ="greater")

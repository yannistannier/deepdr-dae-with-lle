library(readr)
library(mclust)
library(aricode)

args = commandArgs(trailingOnly=TRUE)


x <- c("~/Developpement/cours/DeepReduc", args)
bottle <- read_csv(paste(x, collapse="/"))
labels <- read_csv("~/Developpement/cours/DeepReduc/mnist_label.csv")


nmi=c()
ari=c()
for(i in 1:5){
  clustering=Mclust(bottle,G=10)
  nmi=cbind(nmi,NMI(clustering$classification,as.matrix(labels)[,1]))
  ari=cbind(ari,ARI(clustering$classification,as.matrix(labels)[,1]))
}

NMIPCA=as.vector(nmi);
print(mean(NMIPCA))
ARIPCA=as.vector(ari);
print(mean(ARIPCA))

# print("yess")

# print(args)
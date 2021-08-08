#!/usr/bin/env Rscript
source('r_utils.R')
args = commandArgs(trailingOnly=TRUE)
filename = args[1]
print(filename)
filepath = paste("Data/", filename, ".txt", sep = "")

adj = as.matrix(read.table(filepath, sep="\t"))
normed = KRnorm(adj)
new_name = paste("Data/", filename, "_KR_normed.txt", sep = "")
write.table(normed, file= new_name, row.names=FALSE, col.names=FALSE)

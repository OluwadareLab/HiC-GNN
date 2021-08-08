import numpy as np 
import torch 
from torch_geometric.data import Data 
from torch_sparse import SparseTensor
import networkx as nx
from scipy.linalg import orthogonal_procrustes
import pdb


def convert_to_matrix(adj):
  temp1 = adj[:,0]
  temp2 = adj[:,1]
  temp3 = np.concatenate((temp1, temp2))
  idx = np.unique(temp3)
  size = len(idx)
  mat = np.zeros((size, size))
  for k in range(len(adj)):
    i = int(np.argwhere(adj[k, 0] == idx))
    j = int(np.argwhere(adj[k, 1] == idx))
    mat[i, j] = adj[k,2]
  mat = np.triu(mat) + np.tril(mat.T, 1)
  idx = np.argwhere(np.all(mat[..., :] == 0, axis=0))
  mat = np.delete(mat, idx, axis = 1)
  mat = np.delete(mat , idx, axis = 0)

  return mat


def load_input(input, features):
  adj_mat = input
  if adj_mat.shape[1] == 3:
    adj_mat = convert_to_matrix(adj_mat)
  np.fill_diagonal(adj_mat,0)
  truth = adj_mat
  truth = torch.tensor(truth,dtype=torch.double)
  graph = nx.from_numpy_matrix(adj_mat).to_undirected()
  num_nodes = adj_mat.shape[0]
  edges = list(graph.edges(data=True))
  edge_index = np.empty((len(edges),2))
  edge_weights = np.empty((len(edges)))
  nodes = np.empty(num_nodes)

  for i in range(len(edges)):
    edge_index[i] = np.asarray(edges[i][0:2])
    edge_weights[i] = np.asarray(edges[i][2]["weight"])

  for i in range(num_nodes):
    nodes[i] = np.asarray(i)

  edge_index = torch.tensor(edge_index, dtype=torch.long)
  edge_weights = torch.tensor(edge_weights, dtype=torch.float)
  nodes = torch.tensor(nodes, dtype=torch.long)
  node_attr = torch.tensor(features)

  edge_index=edge_index.t().contiguous()

  
  mask = edge_index[0] != edge_index[1]
  #inv_mask = ~mask

  edge_index = edge_index[:, mask]
  edge_attr = edge_weights[mask]

  data = Data(x=node_attr,edge_index = edge_index, edge_attr = edge_attr, y = truth)

  #device = torch.device('cuda:0')
  #data = data.to(device)
 
  adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1] , value= data.edge_attr, sparse_sizes=(num_nodes, num_nodes))
  data.edge_index = adj.to_symmetric()
  data.edge_attr = None
  return data

def cont2dist(adj,factor):
  dist = (1/adj)**factor
  dist.fill_diagonal_(0)
  max = torch.max(torch.nan_to_num(dist,posinf=0))
  dist = torch.nan_to_num(dist,posinf=max)
  return dist/max

def domain_alignment(list1, list2, embeddings1, embeddings2):
  idx1 = np.unique(list1[:,0]).astype(int)
  diff1 = min(idx1[1:] - idx1[:-1])

  idx2 = np.unique(list2[:,0]).astype(int)
  diff2 = min(idx2[1:] - idx2[:-1])
  
  bins = (diff1/(2*diff2)).astype(int)

  A_list = []
  B_list = []

  for i in range(bins+1):
    Aidx = np.where(np.isin(idx2 + i*diff2, idx1))[0]
    Bidx = np.where(np.isin(idx1, idx2 + i*diff2))[0]

    A_list.append(embeddings2[Aidx,:])
    B_list.append(embeddings1[Bidx,:])


  A = np.concatenate(tuple(A_list))
  B = np.concatenate(tuple(B_list))

  transform = orthogonal_procrustes(A, B)[0]
  fitembed = np.matmul(embeddings2, transform)

  return fitembed

def WritePDB(positions, pdb_file, ctype = "0"):
  '''Save the result as a .pdb file'''
  o_file = open(pdb_file, "w")
  o_file.write("\n")

  col1 = "ATOM"
  col3 = "CA MET"
  col8 = "0.20 10.00"

  bin_num = len(positions)

  for i in range(1, bin_num+1):
      col2 = str(i)
      col4 = "B"+col2
      col5 = "%.3f" % positions[i-1][0]
      col6 = "%.3f" % positions[i-1][1]
      col7 = "%.3f" % positions[i-1][2]
      col2 = " "*(5 - len(col2)) + col2
      col4 = col4 + " " * (6 - len(col4))
      col5 = " " * (8 - len(col5)) + col5
      col6 = " " * (8 - len(col6)) + col6
      col7 = " " * (8 - len(col7)) + col7

      col = (col1, col2, col3, col4, col5, col6, col7,col8)
      line = "%s  %s   %s %s   %s%s%s  %s\n" % col
      o_file.write(line)
  col1 = "CONECT"
  for i in range(1, bin_num+1):
      col2 = str(i)
      j = i + 1
      if j > bin_num:
          if ctype == "1":
              continue
          #j = 1
      col3 = str(j)

      col2 = " " * (5 - len(col2)) + col2
      col3 = " " * (5 - len(col3)) + col3

      line = "%s%s%s\n" % (col1, col2, col3)
      o_file.write(line)

  o_file.write("END")
  o_file.close()






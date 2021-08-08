import numpy as np
from ge import LINE
import argparse
import networkx as nx
import utils 
import os

if __name__=="__main__": 
    if not(os.path.exists('Data')):
        os.makedirs('Data')
        
    parser = argparse.ArgumentParser(description='Generate embeddings for an input file.')
    parser.add_argument('filepath', type=str, help='Input text file path. The file should be either a tab-delimeted interaction frequency matrix, or a  tab-delimeted\
        coordinate list of the form [position1, position2, interaction_frequency].')
    parser.add_argument('-bs', '--batchsize',  type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs used for embeddings generation')

    args = parser.parse_args()
    filepath = args.filepath
    batch_size = args.batchsize
    epochs = args.epochs
    adj = np.loadtxt(filepath)

    if adj.shape[1] == 3:
        adj = utils.convert_to_matrix(adj)

    np.fill_diagonal(adj, 0)

    G = nx.from_numpy_matrix(adj)

    embed = LINE(G,embedding_size=512,order='second')
    embed.train(batch_size=batch_size,epochs=epochs,verbose=1)
    embeddings = embed.get_embeddings()
    embeddings = list(embeddings.values())
    embeddings = np.asarray(embeddings)

    name = os.path.splitext(os.path.basename(filepath))[0]
    np.savetxt('Data/' + name + '_embeddings.txt', embeddings)
    print('Created embeddings corresponding to ' + filepath + ' as Data/' + name + '_embeddings.txt')

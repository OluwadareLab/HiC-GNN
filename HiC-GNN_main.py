import numpy as np
from ge import LINE
import sys
import utils
import networkx as nx 
import os
from models import Net
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr
import ast 
import argparse
import sys
import pdb

if __name__ == "__main__":
    if not(os.path.exists('Outputs')):
        os.makedirs('Outputs')

    if not(os.path.exists('Data')):
        os.makedirs('Data')

    parser = argparse.ArgumentParser(description='Generate embeddings and train a HiC-GNN model.')
    parser.add_argument('filepath', type=str, help='Input text file path. The file should be either a tab-delimeted interaction frequency matrix, or a  tab-delimeted\
    coordinate list of the form [position1, position2, interaction_frequency].')
    parser.add_argument('-c', '--conversions',  type=str, default = '[.1,.1, 2]', help='List of conversion constants of the form [lowest, interval, highest] for an set of\
        equally spaced conversion factors, or of the form [conversion] for a single conversion factor.')
    parser.add_argument('-bs', '--batchsize',  type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs used for embeddings generation')
    parser.add_argument('-lr', '--learningrate',  type=float, default=.001, help='Learning rate for training GCNN.')
    parser.add_argument('-th', '--threshold', type=float, default=1e-8, help='Loss threshold for training termination.')
    args = parser.parse_args()

    filepath = args.filepath
    conversions = args.conversions
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    thresh = args.threshold

    conversions = ast.literal_eval(conversions)

    if len(conversions) == 3:
        conversions = list(np.arange(conversions[0], conversions[2], conversions[1]))
    elif len(conversions) == 1:
        conversions = [conversions[0]]
    else:
        raise Exception('Invalid conversion input.')
        sys.exit(2)   

    name = os.path.splitext(os.path.basename(filepath))[0]

    adj = np.loadtxt(filepath)

    if adj.shape[1] == 3:
        print('Converting coordinate list format to matrix.')
        adj = utils.convert_to_matrix(adj)

    np.fill_diagonal(adj, 0)
    np.savetxt('Data/' + name + '_matrix.txt', adj, delimiter='\t')
    os.system('Rscript normalize.R ' + name + '_matrix')
    print('Created normalized matrix form of ' + filepath + ' as Data/' + name + '_matrix_KR_normed.txt')
    normed = np.loadtxt('Data/' + name + '_matrix_KR_normed.txt')

    G = nx.from_numpy_matrix(adj)

    embed = LINE(G,embedding_size=512,order='second')
    embed.train(batch_size=batch_size,epochs=epochs,verbose=1)
    embeddings = embed.get_embeddings()
    embeddings = list(embeddings.values())
    embeddings = np.asarray(embeddings)

    data = utils.load_input(normed ,embeddings)
    np.savetxt('Data/' + name + '_embeddings.txt', embeddings)
    print('Created embeddings corresponding to ' + filepath + ' as Data/' + name + '_embeddings.txt')

    tempmodels = []
    tempspear = []
    tempmse = []
    model_list = []

    for conversion in conversions:
        print(f'Training model using conversion value {conversion}.')
        model = Net()
        criterion = MSELoss()
        optimizer = Adam(
        model.parameters(), lr=lr)

        oldloss = 1 
        lossdiff = 1

        truth = utils.cont2dist(data.y, .5)

        while lossdiff > thresh:
            model.train()
            optimizer.zero_grad() 
            out = model(data.x, data.edge_index)
            loss = criterion(out.float(), truth.float())
            # ===================backward====================
            lossdiff = abs(oldloss - loss)
            loss.backward()
            optimizer.step()
            oldloss = loss
            print(f'Loss: {loss}', end='\r')

        idx = torch.triu_indices(data.y.shape[0],data.y.shape[1],offset=1)
        dist_truth = truth[idx[0,:],idx[1,:]]
        coords = model.get_model(data.x,data.edge_index)
        out = torch.cdist(coords,coords)
        dist_out = out[idx[0,:],idx[1,:]]
        SpRho = spearmanr(dist_truth,dist_out.detach().numpy())[0]

        tempspear.append(SpRho)
        tempmodels.append(coords)
        tempmse.append(loss)
        model_list.append(model)

    idx = tempspear.index(max(tempspear))
    repmod = tempmodels[idx]
    repspear = tempspear[idx]
    repmse = tempmse[idx]
    repconv = conversions[idx]
    repnet = model_list[idx]   

    print(f'Optimal conversion factor: {repconv}')
    print(f'Optimal dSCC: {repspear}')

    with open('Outputs/' + name + '_log.txt', 'w') as f:
        line1 = f'Optimal conversion factor: {repconv}\n'
        line2 = f'Optimal dSCC: {repspear}\n'
        line3 = f'Final MSE loss: {repmse}\n'
        f.writelines([line1, line2, line3])


    torch.save(repnet.state_dict(), 'Outputs/' + name + '_weights.pt')
    print('Saved trained model corresponding to ' + filepath + ' to Outputs/' + name + '_weights.pt')

    utils.WritePDB(repmod*100, 'Outputs/' + name + '_structure.pdb')
    print('Saved optimal structure corresponding to ' + filepath + ' to Outputs/' + name + '_structure.pdb')

    



    








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
import pdb 
import sys

if __name__ == "__main__":
    if not(os.path.exists('Outputs')):
        os.makedirs('Outputs')

    if not(os.path.exists('Data')):
        os.makedirs('Data')

    parser = argparse.ArgumentParser(description='Generalize a trained model to new data. If the system can not find embeddings and normalized matrices\
    corresponding to the base names of list_trained and list_untrained in the Data directory, the script automatically generates them. If the system cannot\
    find model weights corresponding to the base name of list_trained in the Outputs directory, the script automatically trains a model using the data and embeddings\
    corresponding to the data in list_trained.')
    parser.add_argument('list_trained', type=str, help='File path for list format of raw Hi-C corresponding to embeddings_trained.')
    parser.add_argument('list_untrained', type=str, help='File path for list format of raw Hi-C corresponding to embeddings_untrained.')
    parser.add_argument('-c', '--conversions',  type=str, default = '[.1,.1, 2]', help='List of conversion constants of the form [lowest, interval, highest] for an set of\
    equally spaced conversion factors, or of the form [conversion] for a single conversion factor.')
    parser.add_argument('-bs', '--batchsize',  type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs used for embeddings generation')
    parser.add_argument('-lr', '--learningrate',  type=float, default=.001, help='Learning rate for training GCNN.')
    parser.add_argument('-th', '--threshold', type=float, default=1e-8, help='Loss threshold for training termination.')
    

    args = parser.parse_args()

    filepath_trained = args.list_trained
    filepath_untrained = args.list_untrained
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

    name_trained = os.path.splitext(os.path.basename(filepath_trained))[0]
    name_untrained = os.path.splitext(os.path.basename(filepath_untrained))[0]

    list_trained = np.loadtxt(filepath_trained)
    list_untrained = np.loadtxt(filepath_untrained)

    if not(os.path.isfile('Data/' + name_trained + '_matrix.txt')):
        print('Failed to find matrix form of ' + filepath_trained + ' from Data/' + name_trained + '_matrix.txt.')
        adj_trained = utils.convert_to_matrix(list_trained)
        np.fill_diagonal(adj_trained, 0)
        np.savetxt('Data/' + name_trained + '_matrix.txt', adj_trained, delimiter='\t')
        print('Created matrix form of ' + filepath_trained + ' as Data/' + name_trained + '_matrix.txt.')
    matrix_trained = np.loadtxt('Data/' + name_trained + '_matrix.txt')

    if not(os.path.isfile('Data/' + name_untrained + '_matrix.txt')):
        print('Failed to find matrix form of ' + filepath_untrained + ' from Data/' + name_untrained + '_matrix.txt.')
        adj_untrained = utils.convert_to_matrix(list_untrained)
        np.fill_diagonal(adj_untrained, 0)
        np.savetxt('Data/' + name_untrained + '_matrix.txt', adj_untrained, delimiter='\t')
        print('Created matrix form of ' + filepath_untrained + ' as Data/' + name_untrained + '_matrix.txt.')
    matrix_untrained = np.loadtxt('Data/' + name_untrained + '_matrix.txt')

    if not(os.path.isfile('Data/' + name_trained + '_matrix_KR_normed.txt')):
        print('Failed to find normalized matrix form of ' + filepath_trained + ' from Data/' + name_trained + '_matrix_KR_normed.txt')
        os.system('Rscript normalize.R ' + name_trained + '_matrix')
        print('Created normalized matrix form of ' + filepath_trained + ' as Data/' + name_trained + '_matrix_KR_normed.txt')
    normed_trained = np.loadtxt('Data/' + name_trained + '_matrix_KR_normed.txt')

    if not(os.path.isfile('Data/' + name_untrained + '_matrix_KR_normed.txt')):
        print('Failed to find normalized matrix form of ' + filepath_untrained + ' from Data/' + name_untrained + '_matrix_KR_normed.txt')
        os.system('Rscript normalize.R ' + name_untrained + '_matrix')
        print('Created normalized matrix form of ' + filepath_untrained + ' as Data/' + name_untrained + '_matrix_KR_normed.txt')
    normed_untrained = np.loadtxt('Data/' + name_untrained + '_matrix_KR_normed.txt')

    if not(os.path.isfile('Data/' + name_trained + '_embeddings.txt')):
        print('Failed to find embeddings corresponding to ' + filepath_trained + ' from Data/' + name_trained + '_embeddings.txt')
        G = nx.from_numpy_matrix(matrix_trained)

        embed_trained = LINE(G,embedding_size=512,order='second')
        embed_trained.train(batch_size=batch_size,epochs=epochs,verbose=1)
        embeddings_trained = embed_trained.get_embeddings()
        embeddings_trained = list(embeddings_trained.values())
        embeddings_trained = np.asarray(embeddings_trained)

        np.savetxt('Data/' + name_trained + '_embeddings.txt', embeddings_trained)
        print('Created embeddings corresponding to ' + filepath_trained + ' as Data/' + name_trained + '_embeddings.txt')
    embeddings_trained = np.loadtxt('Data/' + name_trained + '_embeddings.txt')

    if not(os.path.isfile('Data/' + name_untrained + '_embeddings.txt')):
        print('Failed to find embeddings corresponding to ' + filepath_untrained + ' from Data/' + name_untrained + '_embeddings.txt')
        G = nx.from_numpy_matrix(matrix_untrained)

        embed_untrained = LINE(G,embedding_size=512,order='second')
        embed_untrained.train(batch_size=batch_size,epochs=epochs,verbose=1)
        embeddings_untrained = embed_untrained.get_embeddings()
        embeddings_untrained = list(embeddings_untrained.values())
        embeddings_untrained = np.asarray(embeddings_untrained)

        np.savetxt('Data/' + name_untrained + '_embeddings.txt', embeddings_untrained)
        print('Created embeddings corresponding to ' + filepath_untrained + ' as Data/' + name_untrained + '_embeddings.txt')
    embeddings_untrained = np.loadtxt('Data/' + name_untrained + '_embeddings.txt')

    
    data_trained = utils.load_input(normed_trained , embeddings_trained)
    data_untrained = utils.load_input(normed_untrained , embeddings_untrained)

    if not(os.path.isfile('Outputs/' + name_trained + '_weights.pt')):
        print('Failed to find model weights corresponding to ' + filepath_trained + ' from Outputs/' + name_trained + '_weights.pt')
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

            truth = utils.cont2dist(data_trained.y, conversion)

            while lossdiff > thresh:
                model.train()
                optimizer.zero_grad() 
                out = model(data_trained.x.float(), data_trained.edge_index)
                loss = criterion(out.float(), truth.float())
                # ===================backward====================
                lossdiff = abs(oldloss - loss)
                loss.backward()
                optimizer.step()
                oldloss = loss
                print(f'Loss: {loss}', end='\r')

            idx = torch.triu_indices(data_trained.y.shape[0],data_trained.y.shape[1],offset=1)
            dist_truth = truth[idx[0,:],idx[1,:]]
            coords = model.get_model(data_trained.x.float(),data_trained.edge_index)
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

        with open('Outputs/' + name_trained + '_log.txt', 'w') as f:
            line1 = f'Optimal conversion factor: {repconv}\n'
            line2 = f'Optimal dSCC: {repspear}\n'
            line3 = f'Final MSE loss: {repmse}\n'
            f.writelines([line1, line2, line3])

        torch.save(repnet.state_dict(), 'Outputs/' + name_trained + '_weights.pt')

        utils.WritePDB(repmod*100, 'Outputs/' + name_trained + '_structure.pdb')

        print('Saved trained model corresponding to ' + filepath_trained + ' to Outputs/' + name_trained + '_weights.pt')
        print('Saved optimal structure corresponding to ' + filepath_trained + ' to Outputs/' + name_trained + '_structure.pdb')
        conversions = [repconv]
    
    model = Net()
    model.load_state_dict(torch.load('Outputs/' + name_trained + '_weights.pt'))
    model.eval()

    fitembed = utils.domain_alignment(list_trained, list_untrained, embeddings_trained, embeddings_untrained)
    
    data_untrained_fit = utils.load_input(normed_untrained, fitembed)

    temp_spear = []
    temp_models = []

    for conversion in conversions:
        truth = utils.cont2dist(data_untrained_fit.y,conversion).float()

        idx = torch.triu_indices(data_untrained_fit.y.shape[0], data_untrained_fit.y.shape[1],offset=1)
        dist_truth = truth[idx[0,:],idx[1,:]].detach().numpy()
        coords = model.get_model(data_untrained_fit.x.float(), data_untrained_fit.edge_index)
        out = torch.cdist(coords, coords)
        dist_out = out[idx[0,:],idx[1,:]].detach().numpy()

        SpRho = spearmanr(dist_truth, dist_out)[0]
        temp_spear.append(SpRho)
        temp_models.append(coords)

    idx = temp_spear.index(max(temp_spear))
    repspear = temp_spear[idx]
    repconv = conversions[idx]
    repmod = temp_models[idx]

    print(f'Optimal conversion factor for generalized data: {repconv}')
    print(f'Optimal dSCC for generalized data: {repspear}')


    utils.WritePDB(repmod*100, 'Outputs/' + name_untrained + '_generalized_structure.pdb')
    print('Saved optimal, generalized structure corresponding to ' + filepath_untrained + ' to Outputs/' + name_trained + '_structure.pdb')

    with open('Outputs/' + name_untrained + '_generalized_log.txt', 'w') as f:
            line1 = f'Optimal conversion factor: {repconv}\n'
            line2 = f'Optimal dSCC: {repspear}\n'
            f.writelines([line1, line2])

        




    

    


    


    
    

    

import gc
import os
import random
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch as t
from torch import optim

import MKGCN
from clac_metric import get_metrics
from loss import Myloss
from utils import Sizes, constructHNet, constructNet, get_edge_index

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        folder_path = os.path.dirname(fileN)
        os.makedirs(folder_path, exist_ok=True)
        with open(fileN, 'a') as f:
            pass

        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def train(model, train_data, optimizer, sizes):
    model.train() 
    regression_crit = Myloss()

    def train_epoch():
        model.zero_grad() 
        score = model(train_data)
        loss = regression_crit(train_data['Y_train'], score, model.drug_l, model.target_l, model.alpha1,
                               model.alpha2, sizes)
        model.alpha1 = t.mm(
            t.mm((t.mm(model.drug_k, model.drug_k) + model.lambda1 * model.drug_l).inverse(), model.drug_k),
            2 * train_data['Y_train'] - t.mm(model.alpha2.T, model.target_k.T)).detach() 
        model.alpha2 = t.mm(t.mm((t.mm(model.target_k, model.target_k) + model.lambda2 * model.target_l).inverse(), model.target_k),
                            2 * train_data['Y_train'].T - t.mm(model.alpha1.T, model.drug_k.T)).detach() 
        loss = loss.requires_grad_() 
        loss.backward() 
        optimizer.step() 
        return loss

    for epoch in range(1, sizes.epoch + 1):
        train_reg_loss = train_epoch()
        print("epoch : %d, loss:%.2f" % (epoch, train_reg_loss.item()))
    pass


def PredictScore(train_drug_target_matrix, drug_matrix, target_matrix, seed, sizes):
    np.random.seed(seed)
    train_data = {}
    train_data['Y_train'] = t.DoubleTensor(train_drug_target_matrix) 
    Heter_adj = constructHNet(train_drug_target_matrix, drug_matrix, target_matrix)
    Heter_adj = t.FloatTensor(Heter_adj) 
    Heter_adj_edge_index = get_edge_index(Heter_adj) 
    train_data['Adj'] = {'data': Heter_adj, 'edge_index': Heter_adj_edge_index}

    X = constructNet(train_drug_target_matrix)
    X = t.FloatTensor(X)
    train_data['feature'] = X 
    model = MKGCN.Model(sizes, drug_matrix, target_matrix) 

    optimizer = optim.Adam(model.parameters(), lr=sizes.learn_rate) 
    train(model, train_data, optimizer, sizes) 
    return model(train_data) 


def random_index(index_matrix, sizes): 
    association_nam = index_matrix.shape[1] 
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)
    random.shuffle(random_index) 
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds) 
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp


def crossval_index(drug_target_matrix, sizes): 
    random.seed(sizes.seed)
    pos_index_matrix = np.mat(np.where(drug_target_matrix == 1)) 
    neg_index_matrix = np.mat(np.where(drug_target_matrix == 0))

    pos_index = random_index(pos_index_matrix, sizes)
    equal_neg_samples = np.random.choice(neg_index_matrix.shape[1], pos_index_matrix.shape[1]*sizes.neg_times, replace=False)
    neg_index_matrix = neg_index_matrix[:, equal_neg_samples]
    neg_index = random_index(neg_index_matrix, sizes)

    index = [pos_index[i] + neg_index[i] for i in range(sizes.k_fold)]

    return index


def cross_validation_experiment(drug_target_matrix, drug_matrix, target_matrix, sizes):
    index = crossval_index(drug_target_matrix, sizes)
    metric = np.zeros((1, 7))
    pre_matrix = np.zeros(drug_target_matrix.shape)
    print("seed=%d, evaluating drug-target...." % (sizes.seed))
    for k in range(sizes.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(drug_target_matrix, copy=True)
        train_matrix[tuple(np.array(index[k]).T)] = 0 
        drug_len = drug_target_matrix.shape[0]
        dis_len = drug_target_matrix.shape[1]
        drug_target_res = PredictScore(
            train_matrix, drug_matrix, target_matrix, sizes.seed, sizes)
        predict_y_proba = drug_target_res.reshape(drug_len, dis_len).detach().numpy() 
        pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)] 
        metric_tmp = get_metrics(drug_target_matrix[tuple(np.array(index[k]).T)],
                                 predict_y_proba[tuple(np.array(index[k]).T)]) 
        print('aupr, auc, f1_score, accuracy, recall, specificity, precision: ')
        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print('aupr, auc, f1_score, accuracy, recall, specificity, precision: ')
    print(metric / sizes.k_fold) 
    metric = np.array(metric / sizes.k_fold)
    return metric, pre_matrix 


if __name__ == "__main__":

   
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data_path = '../data/'
    
    drug_sim = np.load(data_path + 'drug_sim_network.npy')
    target_sim = np.load(data_path + 'target_sim_network.npy')
    drug_target_matrix = np.load(data_path + 'drug_target_associations.npy')
    
    target_sim = target_sim + 0.00000001
    drug_sim = drug_sim + 0.00000001

    print('New Train——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————')
    print('drug_sim_network:')
    print(drug_sim.shape)
    print('target_sim_network:')
    print(target_sim.shape)
    print('drug_target_associations:')
    print(drug_target_matrix.shape)

    sizes = Sizes(drug_sim.shape[0], target_sim.shape[0])
    results = []

    print(list(sizes.__dict__.values()))

    result, pre_matrix = cross_validation_experiment(
        drug_target_matrix, drug_sim, target_sim, sizes)

    print('Overall Performance:')
    print(result.tolist()[0])

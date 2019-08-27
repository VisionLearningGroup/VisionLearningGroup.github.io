import numpy as np
from scipy.io import loadmat
import sys

sys.path.append('../utils/')
from utils.utils import dense_to_one_hot

base_dir = './data'
def load_syn(scale=True, usps=False, all_use=False):
    syn_data = loadmat(base_dir + '/syn_number.mat')
    syn_train = syn_data['train_data']
    syn_test =  syn_data['test_data']
    syn_train = syn_train.transpose(0, 3, 1, 2).astype(np.float32)
    syn_test = syn_test.transpose(0, 3, 1, 2).astype(np.float32)
    syn_labels_train = syn_data['train_label']
    syn_labels_test = syn_data['test_label']

    train_label = syn_labels_train
    inds = np.random.permutation(syn_train.shape[0])
    syn_train = syn_train[inds]
    train_label = train_label[inds]
    test_label = syn_labels_test

    # syn_train = syn_train[:25000]
    # train_label = train_label[:25000]
    # syn_test = syn_test[:9000]
    # test_label = test_label[:9000]
    train_label = dense_to_one_hot(train_label)
    test_label = dense_to_one_hot(test_label)

    print('syn number train X shape->',  syn_train.shape)
    print('syn number train y shape->',  train_label.shape)
    print('syn number test X shape->',  syn_test.shape)
    print('syn number test y shape->', test_label.shape)
    return syn_train, train_label, syn_test, test_label

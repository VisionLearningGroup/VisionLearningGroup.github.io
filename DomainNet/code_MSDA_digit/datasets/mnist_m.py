import numpy as np
from scipy.io import loadmat

base_dir = './data'
def load_mnistm(scale=True, usps=False, all_use=False):
    mnistm_data = loadmat(base_dir + '/mnistm_with_label.mat')
    mnistm_train = mnistm_data['train']
    mnistm_test =  mnistm_data['test']
    mnistm_train = mnistm_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_test = mnistm_test.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_labels_train = mnistm_data['label_train']
    mnistm_labels_test = mnistm_data['label_test']

    train_label = np.argmax(mnistm_labels_train, axis=1)
    inds = np.random.permutation(mnistm_train.shape[0])
    mnistm_train = mnistm_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnistm_labels_test, axis=1)
    
    mnistm_train = mnistm_train[:25000]
    train_label = train_label[:25000]
    mnistm_test = mnistm_test[:9000]
    test_label = test_label[:9000]
    print('mnist_m train X shape->',  mnistm_train.shape)
    print('mnist_m train y shape->',  train_label.shape)
    print('mnist_m test X shape->',  mnistm_test.shape)
    print('mnist_m test y shape->', test_label.shape)
    return mnistm_train, train_label, mnistm_test, test_label

import numpy as np
from scipy.io import loadmat

base_dir = './data'
def load_mnist(scale=True, usps=False, all_use=False):
    mnist_data = loadmat(base_dir + '/mnist_data.mat')
    if scale:
        mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
        mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
    else:
        mnist_train = mnist_data['train_28']
        mnist_test =  mnist_data['test_28']
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
        mnist_train = mnist_train.astype(np.float32)
        mnist_test = mnist_test.astype(np.float32)
        mnist_train = mnist_train.transpose((0, 3, 1, 2))
        mnist_test = mnist_test.transpose((0, 3, 1, 2))
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)
    
    mnist_train = mnist_train[:25000]
    train_label = train_label[:25000]
    mnist_test = mnist_test[:25000]
    test_label = test_label[:25000]
    print('mnist train X shape->',  mnist_train.shape)
    print('mnist train y shape->',  train_label.shape)
    print('mnist test X shape->',  mnist_test.shape)
    print('mnist test y shape->', test_label.shape)

    return mnist_train, train_label, mnist_test, test_label

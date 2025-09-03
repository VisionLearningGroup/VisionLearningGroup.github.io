from scipy.io import loadmat
import numpy as np
import sys

sys.path.append('../utils/')
from utils.utils import dense_to_one_hot
base_dir = './data'
def load_svhn():
    svhn_train = loadmat(base_dir + '/svhn_train_32x32.mat')
    svhn_test = loadmat(base_dir + '/svhn_test_32x32.mat')
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)

    print('svhn train y shape before dense_to_one_hot->', svhn_train['y'].shape)
    svhn_label = dense_to_one_hot(svhn_train['y'])
    print('svhn train y shape after dense_to_one_hot->',svhn_label.shape)
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = dense_to_one_hot(svhn_test['y'])
    svhn_train_im = svhn_train_im[:25000]
    svhn_label = svhn_label[:25000]
    svhn_test_im = svhn_test_im[:9000]
    svhn_label_test = svhn_label_test[:9000]
    print('svhn train X shape->',  svhn_train_im.shape)
    print('svhn train y shape->',  svhn_label.shape)
    print('svhn test X shape->',  svhn_test_im.shape)
    print('svhn test y shape->', svhn_label_test.shape)

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test


# Convert mnist-rot data format and create train/val/test splits

import os
import argparse

import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str)
    args = vars(parser.parse_args())

    train_fn = os.path.join(args['datadir'], 'mnist_all_rotation_normalized_float_train_valid.amat')
    test_fn = os.path.join(args['datadir'], 'mnist_all_rotation_normalized_float_test.amat')

    train_val = np.loadtxt(train_fn)
    test = np.loadtxt(test_fn)

    train_val_data = train_val[:, :-1].reshape(-1, 1, 28, 28)
    train_val_labels = train_val[:, -1]

    test_data = test[:, :-1].reshape(-1, 1, 28, 28)
    test_labels = test[:, -1]

    np.savez(os.path.join(args['datadir'], 'train_all.npz'), data=train_val_data, labels=train_val_labels)
    np.savez(os.path.join(args['datadir'], 'train.npz'), data=train_val_data[:10000], labels=train_val_labels[:10000])
    np.savez(os.path.join(args['datadir'], 'valid.npz'), data=train_val_data[10000:], labels=train_val_labels[10000:])
    np.savez(os.path.join(args['datadir'], 'test.npz'), data=test_data, labels=test_labels)
    
    


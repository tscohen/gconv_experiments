
import ast
import sys
sys.path.append('../../')
import argparse
import logging
import time
import os
import imp
import shutil
import pickle
import subprocess
import numpy as np
from chainer import optimizers, cuda, serializers
from progressbar import ProgressBar
from gconv_experiments.augmentation import rotate_transform_batch
from chainer import Variable


def create_result_dir(modelfn, logme):
    # if args.restart_from is None:
    result_dir = 'MNIST_ROT/results/' + os.path.basename(modelfn).split('.')[0]
    result_dir += '/' + time.strftime('r%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log_fn = '%s/log.txt' % result_dir
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_fn, level=logging.DEBUG)
    logging.info(logme)

    # Create init file so we can import the model module
    f = open(os.path.join(result_dir, '__init__.py'), 'wb')
    f.close()

    return log_fn, result_dir


def get_model_and_optimizer(result_dir, modelfn, opt, opt_kwargs, net_kwargs, gpu):
    model_fn = os.path.basename(modelfn)
    model_name = model_fn.split('.')[0]
    module = imp.load_source(model_name, modelfn)
    Net = getattr(module, model_name)

    dst = '%s/%s' % (result_dir, model_fn)
    if not os.path.exists(dst):
        shutil.copy(modelfn, dst)

    dst = '%s/%s' % (result_dir, os.path.basename(__file__))
    if not os.path.exists(dst):
        shutil.copy(__file__, dst)

    # prepare model
    model = Net(**net_kwargs)
    if gpu >= 0:
        model.to_gpu()

    optimizer = optimizers.__dict__[opt](**opt_kwargs)
    optimizer.setup(model)

    return model, optimizer


def preprocess_mnist_data(train_data, test_data, train_labels, test_labels):

    train_mean = np.mean(train_data)  # compute mean over all pixels make sure equivariance is preserved
    train_data -= train_mean
    test_data -= train_mean
    train_std = np.std(train_data)
    train_data /= train_std
    test_data /= train_std
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    return train_data, test_data, train_labels, test_labels


def train_epoch(train_data, train_labels, model, optimizer, batchsize, transformations, silent, gpu=0, finetune=False):

    N = train_data.shape[0]
    pbar = ProgressBar(0, N)
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0

    for i in range(0, N, batchsize):
        x_batch = train_data[perm[i:i + batchsize]]
        y_batch = train_labels[perm[i:i + batchsize]]

        if transformations is not None:
            if 'rotation' == transformations:
                x_batch = rotate_transform_batch(
                    x_batch,
                    rotation=2 * np.pi
                )

        if gpu >= 0:
            x_batch = cuda.to_gpu(x_batch.astype(np.float32))
            y_batch = cuda.to_gpu(y_batch.astype(np.int32))

        optimizer.zero_grads()
        x = Variable(x_batch)
        t = Variable(y_batch)

        loss, acc = model(x, t, train=True, finetune=finetune)
        if not finetune:
            loss.backward()
            optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * y_batch.size
        sum_accuracy += float(cuda.to_cpu(acc.data)) * y_batch.size
        if not silent:
            pbar.update(i + y_batch.size)

    return sum_loss, sum_accuracy


def validate(test_data, test_labels, model, batchsize, silent, gpu):
    N_test = test_data.shape[0]
    pbar = ProgressBar(0, N_test)
    sum_accuracy = 0
    sum_loss = 0

    for i in range(0, N_test, batchsize):
        x_batch = test_data[i:i + batchsize]
        y_batch = test_labels[i:i + batchsize]

        if gpu >= 0:
            x_batch = cuda.to_gpu(x_batch.astype(np.float32))
            y_batch = cuda.to_gpu(y_batch.astype(np.int32))

        x = Variable(x_batch)
        t = Variable(y_batch)
        loss, acc = model(x, t, train=False)

        sum_loss += float(cuda.to_cpu(loss.data)) * y_batch.size
        sum_accuracy += float(cuda.to_cpu(acc.data)) * y_batch.size
        if not silent:
            pbar.update(i + y_batch.size)

    return sum_loss, sum_accuracy


def train(
    modelfn, trainfn, valfn,
    epochs, batchsize,
    opt, opt_kwargs,
    net_kwargs,
    transformations,
    val_freq,
    save_freq,
    seed,
    gpu,
    silent=False, logme=None):

    # Set the seed
    np.random.seed(seed)

    # Load an pre-process the data
    try:
        datadir = os.environ['DATADIR']
    except KeyError:
        raise RuntimeError('Please set DATADIR environment variable (e.g. in ~/.bashrc) '
                           'to a folder containing the required datasets.')

    train_set = np.load(os.path.join(datadir, trainfn))
    val_set = np.load(os.path.join(datadir, valfn))
    train_data = train_set['data']
    train_labels = train_set['labels']
    val_data = val_set['data']
    val_labels = val_set['labels']
    train_data, val_data, train_labels, val_labels = preprocess_mnist_data(
        train_data, val_data, train_labels, val_labels)

    # create result dir
    log_fn, result_dir = create_result_dir(modelfn, logme)

    # create model and optimizer
    model, optimizer = get_model_and_optimizer(result_dir, modelfn, opt, opt_kwargs, net_kwargs, gpu)

    # get the last commit
    subp = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = subp.communicate()
    commit = out.strip()
    if err.strip():
        logging.error('Subprocess returned %s' % err.strip())
    logging.info('Commit: ' + commit)

    # Get number of parameters
    # if not silent:
    #     print 'Parameter name, shape, size:'
    #     for p in model.params():
    #         print p.name, p.data.shape, p.data.size
    num_params = sum([p.data.size for p in model.params()])
    logging.info('Number of parameters:' + str(num_params))
    if not silent:
        print 'Number of parameters:' + str(num_params)

    n_train = train_data.shape[0]
    n_val = val_data.shape[0]

    logging.info('start training...')

    train_epochs = []
    train_errors = []
    train_losses = []
    train_times = []
    val_epochs = []
    val_errors = []
    val_losses = []
    val_times = []

    begin_time = time.time()

    sum_loss, sum_accuracy = validate(val_data, val_labels, model, batchsize, silent, gpu)
    val_times.append(time.time() - begin_time)
    val_epochs.append(0)
    val_errors.append(1. - sum_accuracy / n_val)
    val_losses.append(sum_loss / n_val)
    msg = 'epoch:{:02d}\ttest mean loss={}, error={}'.format(
        0, sum_loss / n_val, 1. - sum_accuracy / n_val)
    logging.info(msg)
    if not silent:
        print '\n%s' % msg

    # learning loop
    for epoch in range(1, epochs + 1):

        sum_loss, sum_accuracy = train_epoch(
            train_data, train_labels, model, optimizer,
            batchsize, transformations, silent, gpu)
        train_times.append(time.time() - begin_time)
        train_epochs.append(epoch)
        train_errors.append(1. - sum_accuracy / n_train)
        train_losses.append(sum_loss / n_train)
        msg = 'epoch:{:02d}\ttrain mean loss={}, error={}'.format(
            epoch, sum_loss / n_train, 1. - sum_accuracy / n_train)
        logging.info(msg)
        if not silent:
            print '\n%s' % msg

        if epoch % val_freq == 0:
            print 'FINETUNING'
            model.start_finetuning()
            sum_loss, sum_accuracy = train_epoch(
                    train_data, train_labels, model, optimizer,
                    batchsize, transformations, silent, gpu, finetune=True)
            msg = 'epoch:{:02d}\tfinetune mean loss={}, error={}'.format(
                epoch, sum_loss / n_train, 1. - sum_accuracy / n_train)
            logging.info(msg)
            if not silent:
                print '\n%s' % msg

            sum_loss, sum_accuracy = validate(val_data, val_labels, model, batchsize, silent, gpu)
            val_times.append(time.time() - begin_time)
            val_epochs.append(epoch)
            val_errors.append(1. - sum_accuracy / n_val)
            val_losses.append(sum_loss / n_val)
            msg = 'epoch:{:02d}\ttest mean loss={}, error={}'.format(
                epoch, sum_loss / n_val, 1. - sum_accuracy / n_val)
            logging.info(msg)
            if not silent:
                print '\n%s' % msg

            mean_error = 1.0 - sum_accuracy / n_val

        if save_freq > 0 and epoch % save_freq == 0:
            print 'Saving model...'
            serializers.save_hdf5(os.path.join(result_dir, 'epoch.' + str(epoch) + '.model'), model)

    print 'Saving model...'
    serializers.save_hdf5(os.path.join(result_dir, 'final.model'), model)

    resdict = {
        'train_times': train_times, 'train_epochs': train_epochs,
        'train_errors': train_errors, 'train_losses': train_losses,
        'val_times': val_times, 'val_epochs': val_epochs,
        'val_errors': val_errors, 'val_losses': val_losses
    }

    print 'Saving results...'
    with open(os.path.join(result_dir, 'results.pickle'), 'wb') as handle:
        pickle.dump(resdict, handle)

    return mean_error, model, resdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfn', type=str, default='models/SE2ConvMR.py')
    parser.add_argument('--trainfn', type=str, default='mnist-rot/train.npz')
    parser.add_argument('--valfn', type=str, default='mnist-rot/valid.npz')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=128)

    parser.add_argument('--opt', type=str, default='Adam', choices=['MomentumSGD', 'Adam', 'AdaGrad', 'RMSprop', 'NesterovAG'])
    parser.add_argument('--opt_kwargs', type=ast.literal_eval, default={})  # {'alpha': 0.001})

    parser.add_argument('--net_kwargs', type=ast.literal_eval, default={})

    # parser.add_argument('--weight_decay', type=float, default=0.0005)
    # parser.add_argument('--alpha', type=float, default=0.001)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--lr_decay_freq', type=int, default=10000)
    # parser.add_argument('--lr_decay_ratio', type=float, default=0.1)

    # parser.add_argument('--restart_from', type=str)
    # parser.add_argument('--epoch_offset', type=int, default=0)

    # parser.add_argument('--flip', type=int, default=0)
    # parser.add_argument('--rot', type=int, default=0)
    # parser.add_argument('--shift', type=int, default=0)
    parser.add_argument('--transformations', type=str, default='')  # ast.literal_eval, default={})

    # parser.add_argument('--size', type=int, default=28)

    parser.add_argument('--val_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    # parser.add_argument('--snapshot_freq', type=int, default=10)

    args = parser.parse_args()

    val_error, model, resdict = train(logme=vars(args), **vars(args))

    print 'Finished training'
    print 'Final validation error:', val_error
    print 'Saving model...'
    import chainer.serializers as sl
    sl.save_hdf5('./my.model', model)


import ast
import argparse
import logging
import time
import os
import imp
import shutil
import subprocess

import numpy as np

import chainer
from chainer import optimizers, cuda, serializers, Variable
import cupy as cp

from progressbar import ProgressBar

from gconv_experiments.augmentation import rotate_transform_batch, hflip_transform_batch,\
    dihedral_transform_batch, translate_transform_batch
from cifar10 import get_cifar10_data


def create_result_dir(resultdir, modelfn):
    result_dir = os.path.join(resultdir, os.path.basename(modelfn).split('.')[0], time.strftime('r%Y_%m_%d_%H_%M_%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log_fn = '%s/log.txt' % result_dir
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_fn, level=logging.DEBUG)

    # Print logs to stderr as well
    logging.getLogger().addHandler(logging.StreamHandler())

    # Create init file so we can import the model module
    # f = open(os.path.join(result_dir, '__init__.py'), 'wb')
    # f.close()

    return log_fn, result_dir


def get_model_and_optimizer(result_dir, modelfn, opt, opt_kwargs, net_kwargs, gpu):
    model_fn = os.path.basename(modelfn)
    model_name = model_fn.split('.')[0]
    module = imp.load_source(model_name, modelfn)
    net = getattr(module, model_name)

    # Copy model definition and this train script to the result dir
    dst = '%s/%s' % (result_dir, model_fn)
    if not os.path.exists(dst):
        shutil.copy(modelfn, dst)
    dst = '%s/%s' % (result_dir, os.path.basename(__file__))
    if not os.path.exists(dst):
        shutil.copy(__file__, dst)

    # Create model
    model = net(**net_kwargs)
    if gpu >= 0:
        model.to_gpu(gpu)

    # Create optimizer
    optimizer = optimizers.__dict__[opt](**opt_kwargs)
    optimizer.setup(model)

    return model, optimizer


def do_epoch(data, labels, model, optimizer, batchsize, transformations, silent, train=True, gpu=0, finetune=False):

    N = data.shape[0]
    pbar = ProgressBar(0, N)
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0

    for i in range(0, N, batchsize):
        x_batch = data[perm[i:i + batchsize]]
        y_batch = labels[perm[i:i + batchsize]]

        if transformations is not None and train:
            if 'rotation' == transformations:
                x_batch = rotate_transform_batch(
                    x_batch,
                    rotation=2 * np.pi
                )

            if 'hflip' == transformations:
                x_batch = hflip_transform_batch(x_batch)

            if 'translate_hflip' == transformations:
                x_batch = hflip_transform_batch(x_batch)
                x_batch = translate_transform_batch(x_batch)

            if 'translate_dihedral' == transformations:
                x_batch = dihedral_transform_batch(x_batch)
                x_batch = translate_transform_batch(x_batch)

        if gpu >= 0:
            x_batch = cuda.to_gpu(x_batch.astype(np.float32), device=gpu)
            y_batch = cuda.to_gpu(y_batch.astype(np.int32), device=gpu)

        x = Variable(x_batch)
        t = Variable(y_batch)

        if train:
            optimizer.zero_grads()

        loss, acc = model(x, t, train=train, finetune=finetune)

        if train:
            loss.backward()
            optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * y_batch.size
        sum_accuracy += float(cuda.to_cpu(acc.data)) * y_batch.size
        if not silent:
            pbar.update(i + y_batch.size)

    return sum_loss, sum_accuracy


def train(
    datadir,
    resultdir,
    modelfn, trainfn, valfn,
    epochs, batchsize,
    opt, opt_kwargs,
    net_kwargs,
    transformations,
    val_freq,
    save_freq,
    weight_decay,
    lr_decay_schedule,
    lr_decay_factor,
    gpu,
    seed,
    silent=False,
    logme=None):

    # Set the seed
    np.random.seed(seed)
    cp.random.seed(seed)

    # Load an pre-process the data
    train_data, train_labels, val_data, val_labels = get_cifar10_data(datadir, trainfn, valfn)

    # Create result dir
    log_fn, resultdir = create_result_dir(resultdir, modelfn)
    logging.info(logme)

    # create model and optimizer
    model, optimizer = get_model_and_optimizer(resultdir, modelfn, opt, opt_kwargs, net_kwargs, gpu)

    if weight_decay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

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
    num_params = sum([p.data.size for p in model.params()])
    logging.info('Number of parameters:' + str(num_params))

    num_train = train_data.shape[0]
    num_val = val_data.shape[0]

    # Learning rate decay
    if lr_decay_factor > 0:
        lr_decay_schedule = [int(istr) for istr in lr_decay_schedule.split('-')]
        logging.info('Using decay schedule: ' + str(lr_decay_schedule) + ' ' + str(lr_decay_factor))

    logging.info('start training...')

    # learning loop
    for epoch in range(1, epochs + 1):

        sum_loss, sum_accuracy = do_epoch(
            train_data, train_labels, model, optimizer, batchsize, transformations, silent, True, gpu
        )
        msg = '\nepoch:{:02d}\ttrain mean loss={}, error={}'.format(
            epoch, sum_loss / num_train, 1. - sum_accuracy / num_train)
        logging.info(msg)

        if epoch % val_freq == 0:

            # TODO: finetune in last epoch *before* validation epoch?
            logging.info('START FINETUNING')
            model.start_finetuning()
            sum_loss, sum_accuracy = do_epoch(
                train_data, train_labels, model, optimizer, batchsize, transformations, silent, False, gpu, True
            )
            msg = '\nepoch:{:02d}\tfinetune mean loss={}, error={}'.format(
                epoch, sum_loss / num_train, 1. - sum_accuracy / num_train)
            logging.info(msg)

            # sum_loss, sum_accuracy = validate(val_data, val_labels, model, batchsize, silent, gpu)
            sum_loss, sum_accuracy = do_epoch(
                val_data, val_labels, model, None, batchsize, None, silent, False, gpu, False
            )
            msg = '\nepoch:{:02d}\ttest mean loss={}, error={}'.format(
                epoch, sum_loss / num_val, 1. - sum_accuracy / num_val)
            logging.info(msg)

            mean_error = 1.0 - sum_accuracy / num_val

        if save_freq > 0 and epoch % save_freq == 0:
            logging.info('Saving model...')
            serializers.save_hdf5(os.path.join(resultdir, 'epoch.' + str(epoch) + '.model'), model)

        if lr_decay_factor > 0 and epoch in lr_decay_schedule:
            logging.info('Learning rate drop from ' + str(optimizer.lr) + ' to ' + str(optimizer.lr * lr_decay_factor))
            optimizer.lr *= lr_decay_factor

    logging.info('Saving model...')
    serializers.save_hdf5(os.path.join(resultdir, 'final.model'), model)

    return mean_error, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str,
                        default='/var/scratch/tscohen/data/CIFAR10')
    parser.add_argument('--resultdir', type=str,
                        default='/var/scratch/tscohen/results/')

    parser.add_argument('--modelfn', type=str,
                        default='CIFAR10/models/AllCNNC.py')
    parser.add_argument('--trainfn', type=str,
                        default='train_all.npz')
    parser.add_argument('--valfn', type=str,
                        default='test.npz')

    parser.add_argument('--epochs', type=int,
                        default=100)
    parser.add_argument('--batchsize', type=int,
                        default=128)

    parser.add_argument('--opt', type=str, default='MomentumSGD',
                        choices=['MomentumSGD', 'Adam', 'AdaGrad', 'RMSprop', 'NesterovAG'])
    parser.add_argument('--opt_kwargs', type=ast.literal_eval,
                        default={})  # usage: --opt_kwargs="{'lr': 0.05}"

    parser.add_argument('--net_kwargs', type=ast.literal_eval,
                        default={})

    parser.add_argument('--weight_decay', type=float,
                        default=0.001)

    parser.add_argument('--lr_decay_schedule', type=str,
                        default='100-200-300')
    parser.add_argument('--lr_decay_factor', type=float,
                        default=0.)  # default 0 means no learning rate decay

    parser.add_argument('--transformations', type=str,
                        default='')

    parser.add_argument('--val_freq', type=int,
                        default=10)
    parser.add_argument('--save_freq', type=int,
                        default=10)

    parser.add_argument('--gpu', type=int,
                        default=0)
    parser.add_argument('--seed', type=int,
                        default=0)

    args = parser.parse_args()
    vargs = vars(args)

    with cp.cuda.Device(vargs['gpu']):
        val_error, model = train(logme=vargs, **vargs)

    print 'Finished training'
    print 'Final validation error:', val_error
    print 'Saving model...'
    import chainer.serializers as sl
    sl.save_hdf5('./my.model', model)

from train import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfn', type=str, default='mnist-rot/train_all.npz')
    parser.add_argument('--valfn', type=str, default='mnist-rot/test.npz')
    parser.add_argument('--modelpath', type=str, default='MNIST_ROT/models/')
    parser.add_argument('--repeats', type=int, default=5)

    args = vars(parser.parse_args())

    models = ['P4CNN.py', 'P4CNN_RP.py', 'Z2CNN.py']
    model_epochs = {'Z2CNN.py': 300, 'P4CNN.py': 100, 'P4CNN_RP.py': 100}
    augmentations = ['rotation']  # , '']
    errors = {}
    for t in augmentations:
        for m in models:

            errors[m + t] = []
            for i in range(args['repeats']):
                print 'Model:', m
                print 'Augmentation:', t
                print 'Repeat:', i + 1

                modelfn = os.path.join(args['modelpath'], m)
                err, _, _ = train(
                    modelfn=modelfn,
                    trainfn=args['trainfn'],
                    valfn=args['valfn'],
                    epochs=model_epochs[m],
                    batchsize=128,
                    opt='Adam', opt_kwargs={'alpha': 0.001},
                    net_kwargs={},
                    transformations=t,
                    val_freq=25,
                    save_freq=25,
                    seed=i,
                    gpu=0,
                    silent=False,
                    logme=None
                )

                errors[m + t].append(err)

            print '-- Evaluation run complete --'
            print 'Model:', m,
            print 'augmentations:', t
            print 'error rates:', errors[m + t]
            print 'mean error rate:', np.mean(errors[m + t]), '+/-', np.std(errors[m + t])

    print 'EXPERIMENT COMPLETED'
    print errors
    for mt in errors:
        print 'mean error rate for', mt, ':', np.mean(errors[mt]), '+/-', np.std(errors[mt])

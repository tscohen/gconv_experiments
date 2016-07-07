import matplotlib.pyplot as plt
import numpy as np
from groupy.gfunc.plot.plot_z2 import plot_z2
from groupy.gfunc.plot.plot_p4 import plot_p4
from groupy.gfunc.plot.plot_p4m import plot_p4m

# Code used to create the figures in
# T.S. Cohen, M. Welling, Group Equivariant Convolutional Networks.
# Proceedings of the International Conference on Machine Learning (ICML), 2016


def paper_plots_p4():
    import matplotlib
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    im_e, fmaps_e = testplot_p4(r=0)
    im_r, fmaps_r = testplot_p4(r=1)

    plot_p4(fmaps_e, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6))
    plt.savefig('./p4_fmap_e_mini.eps', format='eps', dpi=600)
    plot_p4(fmaps_r, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6))
    plt.savefig('./p4_fmap_r_mini.eps', format='eps', dpi=600)


def gif_p4():

    im_e, fmaps_e = testplot_p4(r=0)
    im_r, fmaps_r = testplot_p4(r=1)
    im_r2, fmaps_r2 = testplot_p4(r=2)
    im_r3, fmaps_r3 = testplot_p4(r=3)

    plot_p4(fmaps_e, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6), rlabels='none')
    plt.savefig('./p4_fmap_e_mini.png', format='png', dpi=200)
    plot_p4(fmaps_r, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6), rlabels='none')
    plt.savefig('./p4_fmap_r_mini.png', format='png', dpi=200)
    plot_p4(fmaps_r2, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6), rlabels='none')
    plt.savefig('./p4_fmap_r2_mini.png', format='png', dpi=200)
    plot_p4(fmaps_r3, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6), rlabels='none')
    plt.savefig('./p4_fmap_r3_mini.png', format='png', dpi=200)

    plt.figure(figsize=(1.6, 1.6))
    plot_z2(im_e)
    plt.axis('off')
    plt.savefig('./f_e.png', format='png', dpi=200)
    plot_z2(im_r)
    plt.savefig('./f_r.png', format='png', dpi=200)
    plot_z2(im_r2)
    plt.savefig('./f_r2.png', format='png', dpi=200)
    plot_z2(im_r3)
    plt.savefig('./f_r3.png', format='png', dpi=200)


def testplot_p4(im=None, r=0):
    if im is None:
        im = np.zeros((5, 5), dtype='float32')
        im[0:5, 1] = 1.
        im[0, 1:4] = 1.
        im[2, 1:3] = 1.

    from groupy.gfunc.z2func_array import Z2FuncArray
    from groupy.garray.C4_array import C4Array
    def rotate_z2_func(im, r):
        imf = Z2FuncArray(im)
        rot = C4Array([r], 'int')
        rot_imf = rot * imf
        return rot_imf.v

    im = rotate_z2_func(im, r)

    filter1 = np.array([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).astype(np.float32)
    filter2 = rotate_z2_func(filter1, 1)
    filter3 = rotate_z2_func(filter1, 2)
    filter4 = rotate_z2_func(filter1, 3)

    from chainer.functions import Convolution2D
    from chainer import Variable
    im = im.astype(np.float32)
    pad = 2
    imf1 = Convolution2D(in_channels=1, out_channels=1, ksize=3, bias=0., pad=pad, initialW=filter1)(
        Variable(im[None, None])).data[0, 0]
    imf2 = Convolution2D(in_channels=1, out_channels=1, ksize=3, bias=0., pad=pad, initialW=filter2)(
        Variable(im[None, None])).data[0, 0]
    imf3 = Convolution2D(in_channels=1, out_channels=1, ksize=3, bias=0., pad=pad, initialW=filter3)(
        Variable(im[None, None])).data[0, 0]
    imf4 = Convolution2D(in_channels=1, out_channels=1, ksize=3, bias=0., pad=pad, initialW=filter4)(
        Variable(im[None, None])).data[0, 0]

    return im, np.r_[[imf1, imf2, imf3, imf4]]


def paper_plots_p4m():
    import matplotlib
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    im_e, fmaps_e = testplot_p4m(m=0, r=0)
    im_r, fmaps_r = testplot_p4m(m=0, r=1)
    im_m, fmaps_m = testplot_p4m(m=1, r=0)

    plot_p4m(fmaps_e.reshape(2, 4, 7, 7), rlabels='cayley2', fontsize=10, labelpad_factor_1=0.2,
             labelpad_factor_2=0.8, labelpad_factor_3=0.5, labelpad_factor_4=1.2,
             figsize=(2.5, 2.5), rcolor='red', mcolor='blue')
    plt.savefig('./p4m_fmap_e_mini.eps', format='eps', dpi=600)
    plot_p4m(fmaps_r.reshape(2, 4, 7, 7), rlabels='cayley2', fontsize=10, labelpad_factor_1=0.2,
             labelpad_factor_2=0.8, labelpad_factor_3=0.5, labelpad_factor_4=1.2,
             figsize=(2.5, 2.5), rcolor='red', mcolor='blue')
    plt.savefig('./p4m_fmap_r_mini.eps', format='eps', dpi=600)
    plot_p4m(fmaps_m.reshape(2, 4, 7, 7), rlabels='cayley2', fontsize=10, labelpad_factor_1=0.2,
             labelpad_factor_2=0.8, labelpad_factor_3=0.5, labelpad_factor_4=1.2,
             figsize=(2.5, 2.5), rcolor='red', mcolor='blue')
    plt.savefig('./p4m_fmap_m_mini.eps', format='eps', dpi=600)


def testplot_p4m(im=None, m=0, r=0):

    if im is None:
        im = np.zeros((5, 5), dtype='float32')
        im[0:5, 1] = 1.
        im[0, 1:4] = 1.
        im[2, 1:3] = 1.

    from groupy.gfunc.z2func_array import Z2FuncArray
    from groupy.garray.D4_array import D4Array
    def rotate_flip_z2_func(im, flip, theta_index):
        imf = Z2FuncArray(im)
        rot = D4Array([flip, theta_index], 'int')
        rot_imf = rot * imf
        return rot_imf.v
    im = rotate_flip_z2_func(im, m, r)

    filter_e = np.array([[-1., -4., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]])

    from groupy.gconv.chainer_gconv.p4m_conv import P4MConvZ2
    from chainer import Variable
    from chainer import cuda

    print im.shape

    imv = Variable(cuda.to_gpu(im.astype('float32').reshape(1, 1, 5, 5)))
    conv = P4MConvZ2(in_channels=1, out_channels=1, ksize=3, pad=2, flat_channels=True, initialW=filter_e.reshape(1, 1, 1, 3, 3))
    conv.to_gpu()
    conv_imv = conv(imv)
    print im.shape, conv_imv.data.shape
    return im, cuda.to_cpu(conv_imv.data)

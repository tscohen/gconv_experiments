
from math import sqrt

from chainer import ChainList
import chainer.functions as F
from chainer.links import Convolution2D

from gconv_experiments.residual_block_2d import ResBlock2D


class ResNet(ChainList):

    def __init__(self, num_blocks=18, nc32=16, nc16=32, nc8=64):
        """
        :param num_blocks: the number of resnet blocks per stage. There are 3 stages, for feature map width 32, 16, 8.
        Total number of layers is 6 * num_blocks + 2
        :param nc32: the number of feature maps in the first stage (where feature maps are 32x32)
        :param nc16: the number of feature maps in the second stage (where feature maps are 16x16)
        :param nc8: the number of feature maps in the third stage (where feature maps are 8x8)
        """
        ksize = 3
        pad = 1
        ws = sqrt(2.)  # This makes the initialization equal to that of He et al.

        super(ResNet, self).__init__()

        # The first layer is always a convolution.
        self.add_link(
            Convolution2D(in_channels=3, out_channels=nc32, ksize=ksize, stride=1, pad=pad, wscale=ws)
        )

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 32x32 feature maps
        for i in range(num_blocks):
            self.add_link(
                ResBlock2D(
                    in_channels=nc32, out_channels=nc32, ksize=ksize,
                    fiber_map='id', stride=1, pad=pad, wscale=ws
                )
            )

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 16x16 feature maps
        # The first convolution uses stride 2
        for i in range(num_blocks):
            stride = 1 if i > 0 else 2
            fiber_map = 'id' if i > 0 else 'linear'
            nc_in = nc16 if i > 0 else nc32
            self.add_link(
                ResBlock2D(
                    in_channels=nc_in, out_channels=nc16, ksize=ksize,
                    fiber_map=fiber_map, stride=stride, pad=pad, wscale=ws
                )
            )

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 8x8 feature maps
        # The first convolution uses stride 2
        for i in range(num_blocks):
            stride = 1 if i > 0 else 2
            fiber_map = 'id' if i > 0 else 'linear'
            nc_in = nc8 if i > 0 else nc16
            self.add_link(
                ResBlock2D(
                    in_channels=nc_in, out_channels=nc8, ksize=ksize,
                    fiber_map=fiber_map, stride=stride, pad=pad, wscale=ws
                )
            )

        # Add BN and final layer
        # We do ReLU and average pooling between BN and final layer,
        # but since these are stateless they don't require a Link.
        self.add_link(F.BatchNormalization(size=nc8))
        self.add_link(Convolution2D(in_channels=nc8, out_channels=10, ksize=1, stride=1, pad=0, wscale=ws))

    def __call__(self, x, t, train=True, finetune=False):

        h = x

        # First conv layer
        h = self[0](h)

        # Residual blocks
        for i in range(1, len(self) - 2):
            h = self[i](h, train, finetune)

        # BN, relu, pool, final layer
        h = self[-2](h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, ksize=h.data.shape[2:])
        h = self[-1](h)
        h = F.reshape(h, h.data.shape[:2])

        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

    def start_finetuning(self):
        for c in self.children():
            if isinstance(c, ResBlock2D):
                c.bn1.start_finetuning()
                c.bn2.start_finetuning()


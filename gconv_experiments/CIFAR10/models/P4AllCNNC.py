import numpy as np
import chainer.functions as F
from chainer import Chain

from groupy.gconv.chainer_gconv.p4_conv import P4ConvZ2, P4ConvP4

from gconv_experiments.conv_bn_act import ConvBNAct


class P4AllCNNC(Chain):

    def __init__(self):
        bn = True
        ksize = 3
        pad = 1
        act = F.relu

        super(P4AllCNNC, self).__init__(

            l1=ConvBNAct(
                conv=P4ConvZ2(in_channels=3, out_channels=48, ksize=ksize, stride=1, pad=pad),
                bn=bn,
                act=act
            ),

            l2=ConvBNAct(
                conv=P4ConvP4(in_channels=48, out_channels=48, ksize=ksize, stride=1, pad=pad),
                bn=bn,
                act=act
            ),

            l3=ConvBNAct(
                conv=P4ConvP4(in_channels=48, out_channels=48, ksize=ksize, stride=2, pad=pad),
                bn=bn,
                act=act
            ),

            l4=ConvBNAct(
                conv=P4ConvP4(in_channels=48, out_channels=96, ksize=ksize, stride=1, pad=pad),
                bn=bn,
                act=act
            ),

            l5=ConvBNAct(
                conv=P4ConvP4(in_channels=96, out_channels=96, ksize=ksize, stride=1, pad=pad),
                bn=bn,
                act=act
            ),

            l6=ConvBNAct(
                conv=P4ConvP4(in_channels=96, out_channels=96, ksize=ksize, stride=2, pad=pad),
                bn=bn,
                act=act
            ),

            l7=ConvBNAct(
                conv=P4ConvP4(in_channels=96, out_channels=96, ksize=ksize, stride=1, pad=pad),
                bn=bn,
                act=act
            ),

            l8=ConvBNAct(
                conv=P4ConvP4(in_channels=96, out_channels=96, ksize=1, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            # Note: it's unusual to have a bn + relu before softmax, but this is what's described by springenberg et al.
            l9=ConvBNAct(
                conv=P4ConvP4(in_channels=96, out_channels=10, ksize=1, stride=1, pad=0),
                bn=bn,
                act=act
            ),
        )

        wtscale = 0.035

        self.l1.conv.W.data = (np.random.randn(*self.l1.conv.W.data.shape) * wtscale).astype(np.float32)
        self.l2.conv.W.data = (np.random.randn(*self.l2.conv.W.data.shape) * wtscale).astype(np.float32)
        self.l3.conv.W.data = (np.random.randn(*self.l3.conv.W.data.shape) * wtscale).astype(np.float32)
        self.l4.conv.W.data = (np.random.randn(*self.l4.conv.W.data.shape) * wtscale).astype(np.float32)
        self.l5.conv.W.data = (np.random.randn(*self.l5.conv.W.data.shape) * wtscale).astype(np.float32)
        self.l6.conv.W.data = (np.random.randn(*self.l6.conv.W.data.shape) * wtscale).astype(np.float32)
        self.l7.conv.W.data = (np.random.randn(*self.l7.conv.W.data.shape) * wtscale).astype(np.float32)
        self.l8.conv.W.data = (np.random.randn(*self.l8.conv.W.data.shape) * wtscale).astype(np.float32)
        self.l9.conv.W.data = (np.random.randn(*self.l9.conv.W.data.shape) * wtscale).astype(np.float32)

    def __call__(self, x, t, train=True, finetune=False):

        h = x
        h = F.dropout(h, ratio=0.2, train=train)
        h = self.l1(h, train, finetune)
        h = self.l2(h, train, finetune)
        h = self.l3(h, train, finetune)
        h = F.dropout(h, ratio=0.5, train=train)
        h = self.l4(h, train, finetune)
        h = self.l5(h, train, finetune)
        h = self.l6(h, train, finetune)
        h = F.dropout(h, ratio=0.5, train=train)
        h = self.l7(h, train, finetune)
        h = self.l8(h, train, finetune)
        h = self.l9(h, train, finetune)

        h = F.sum(h, axis=-1)
        h = F.sum(h, axis=-1)
        h = F.sum(h, axis=-1)
        h /= 8 * 8 * 4

        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

    def start_finetuning(self):
        for c in self.children():
            if isinstance(c, ConvBNAct):
                if c.bn:
                    c.bn.start_finetuning()

import chainer.functions as F
from chainer import Chain

from gconv_experiments.conv_bn_act import ConvBNAct


class Z2CNN(Chain):

    def __init__(self):
        ksize = 3
        bn = True
        act = F.relu
        self.dr = 0.3
        super(Z2CNN, self).__init__(

            l1=ConvBNAct(
                conv=F.Convolution2D(in_channels=1, out_channels=20, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            l2=ConvBNAct(
                conv=F.Convolution2D(in_channels=20, out_channels=20, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            l3=ConvBNAct(
                conv=F.Convolution2D(in_channels=20, out_channels=20, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            l4=ConvBNAct(
                conv=F.Convolution2D(in_channels=20, out_channels=20, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            l5=ConvBNAct(
                conv=F.Convolution2D(in_channels=20, out_channels=20, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            l6=ConvBNAct(
                conv=F.Convolution2D(in_channels=20, out_channels=20, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            top=F.Convolution2D(in_channels=20, out_channels=10, ksize=4, stride=1, pad=0),
        )

    def __call__(self, x, t, train=True, finetune=False):

        h = self.l1(x, train, finetune)
        h = F.dropout(h, self.dr, train)
        h = self.l2(h, train, finetune)

        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0, cover_all=True, use_cudnn=True)

        h = self.l3(h, train, finetune)
        h = F.dropout(h, self.dr, train)
        h = self.l4(h, train, finetune)
        h = F.dropout(h, self.dr, train)
        h = self.l5(h, train, finetune)
        h = F.dropout(h, self.dr, train)
        h = self.l6(h, train, finetune)
        h = F.dropout(h, self.dr, train)

        h = self.top(h)

        h = F.max(h, axis=-1, keepdims=False)
        h = F.max(h, axis=-1, keepdims=False)

        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

    def start_finetuning(self):
        for c in self.children():
            if isinstance(c, ConvBNAct):
                if c.bn:
                    c.bn.start_finetuning()


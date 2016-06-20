
import chainer
import chainer.links as L
import chainer.functions as F


# New style residual block
class ResBlock2D(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, fiber_map='id', conv_link=L.Convolution2D,
                 stride=1, pad=1, wscale=1):

        assert ksize % 2 == 1

        if not pad == (ksize - 1) // 2:
            raise NotImplementedError()

        super(ResBlock2D, self).__init__(
            bn1=L.BatchNormalization(in_channels),
            conv1=conv_link(
                in_channels=in_channels, out_channels=out_channels, ksize=ksize, stride=stride, pad=pad, wscale=wscale),
            bn2=L.BatchNormalization(out_channels),
            conv2=conv_link(
                in_channels=out_channels, out_channels=out_channels, ksize=ksize, stride=1, pad=pad, wscale=wscale)
        )

        if fiber_map == 'id':
            if not in_channels == out_channels:
                raise ValueError('fiber_map cannot be identity when channel dimension is changed.')
            self.fiber_map = F.identity
        elif fiber_map == 'zero_pad':
            raise NotImplementedError()
        elif fiber_map == 'linear':
            fiber_map = conv_link(
                in_channels=in_channels, out_channels=out_channels, ksize=1, stride=stride, pad=0, wscale=wscale)
            self.add_link('fiber_map', fiber_map)
        else:
            raise ValueError('Unknown fiber_map: ' + str(type))

    def __call__(self, x, train, finetune):
        h = self.conv1(F.relu(self.bn1(x, test=not train, finetune=finetune)))
        h = self.conv2(F.relu(self.bn2(h, test=not train, finetune=finetune)))
        hx = self.fiber_map(x)
        return hx + h

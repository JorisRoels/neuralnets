import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Linear layer

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, in_channels, out_channels, dropout=0.0, activation=None, norm=None):
        super().__init__()

        # initialize linear layer
        self.unit = nn.Sequential(nn.Linear(in_channels, out_channels))

        # apply normalization
        if norm == 'batch':
            self.unit.add_module('norm', nn.BatchNorm1d(int(out_channels)))
        elif norm == 'instance':
            self.unit.add_module('norm', nn.InstanceNorm1d(int(out_channels)))

        # apply dropout
        if dropout > 0.0:
            self.unit.add_module('dropout', nn.Dropout(p=dropout))

        # apply activation
        if activation == 'relu':
            self.unit.add_module('activation', nn.ReLU())
        elif activation == 'sigmoid':
            self.unit.add_module('activation', nn.Sigmoid())

    def forward(self, inputs):

        return self.unit(inputs)


class Conv2D(nn.Module):
    """
    2D convolutional layer

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param optional kernel_size: kernel size of the convolutions
    :param optional stride: stride of the convolutions
    :param optional padding: padding used for border cases ("SAME", "VALID" or None)
    :param optional bias: use bias term or not
    :param optional dilation: dilation of the convolution
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='SAME', bias=True, dilation=1,
                 dropout=0.0, activation=None, norm=None):
        super().__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else:  # VALID (no) padding
            p = 0

        # initialize convolutional block
        self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation))

        # apply normalization
        if norm == 'batch':
            self.unit.add_module('norm', nn.BatchNorm2d(int(out_channels)))
        elif norm == 'instance':
            self.unit.add_module('norm', nn.InstanceNorm2d(int(out_channels)))

        # apply dropout
        if dropout > 0.0:
            self.unit.add_module('dropout', nn.Dropout2d(p=dropout))

        # apply activation
        if activation == 'relu':
            self.unit.add_module('activation', nn.ReLU())
        elif activation == 'sigmoid':
            self.unit.add_module('activation', nn.Sigmoid())

    def forward(self, inputs):

        return self.unit(inputs)


class DenseLayer2D(nn.Module):
    """
    2D Dense convolutional layer

    :param in_channels: number of input channels
    :param k: how many filters to add each layer
    :param bn_size: multiplicative factor for number of bottle neck layers
    :param optional kernel_size: kernel size of the convolutions
    :param optional padding: padding used for border cases ("SAME", "VALID" or None)
    :param optional bias: use bias term or not
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, in_channels, k, bn_size, kernel_size=3, padding='SAME', bias=False, dropout=0.0, activation=None,
                 norm=None):
        super().__init__()

        self.conv1 = Conv2D(in_channels, bn_size * k, kernel_size=1, bias=bias, activation=activation, norm=norm)
        self.conv2 = Conv2D(bn_size * k, k, kernel_size=kernel_size, bias=bias, padding=padding, activation=activation,
                            norm=norm, dropout=dropout)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]
        x = torch.cat(x, 1)
        x = self.conv2(self.conv1(x))
        return x


class Conv3D(nn.Module):
    """
    3D convolutional layer

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param optional kernel_size: kernel size of the convolutions
    :param optional stride: stride of the convolutions
    :param optional padding: padding used for border cases ("SAME", "VALID" or None)
    :param optional bias: use bias term or not
    :param optional dilation: dilation of the convolution
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='SAME', bias=True, dilation=1,
                 dropout=0.0, activation=None, norm=None):
        super().__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else:  # VALID (no) padding
            p = 0

        # initialize convolutional block
        self.unit = nn.Sequential(nn.Conv3d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation))

        # apply normalization
        if norm == 'batch':
            self.unit.add_module('norm', nn.BatchNorm3d(int(out_channels)))
        elif norm == 'instance':
            self.unit.add_module('norm', nn.InstanceNorm3d(int(out_channels)))

        # apply dropout
        if dropout > 0.0:
            self.unit.add_module('dropout', nn.Dropout3d(p=dropout))

        # apply activation
        if activation == 'relu':
            self.unit.add_module('activation', nn.ReLU())
        elif activation == 'sigmoid':
            self.unit.add_module('activation', nn.Sigmoid())

    def forward(self, inputs):

        return self.unit(inputs)


class DenseLayer3D(nn.Module):
    """
    3D Dense convolutional layer

    :param in_channels: number of input channels
    :param k: how many filters to add each layer
    :param bn_size: multiplicative factor for number of bottle neck layers
    :param optional kernel_size: kernel size of the convolutions
    :param optional padding: padding used for border cases ("SAME", "VALID" or None)
    :param optional bias: use bias term or not
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, in_channels, k, bn_size, kernel_size=3, padding='SAME', bias=False, dropout=0.0, activation=None,
                 norm=None):
        super().__init__()

        self.conv1 = Conv3D(in_channels, bn_size * k, kernel_size=1, bias=bias, activation=activation, norm=norm)
        self.conv2 = Conv3D(bn_size * k, k, kernel_size=kernel_size, bias=bias, padding=padding, activation=activation,
                            norm=norm, dropout=dropout)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]
        x = torch.cat(x, 1)
        x = self.conv2(self.conv1(x))
        return x


class UNetConvBlock2D(nn.Module):
    """
    2D convolutional block of the classical U-Net (i.e. two sequential convolutional layers)

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param optional kernel_size: kernel size of the convolutions
    :param optional padding: padding used for border cases ("SAME", "VALID" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', norm='batch', dropout=0.0,
                 activation='relu'):
        super().__init__()

        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                            dropout=dropout, activation=activation, norm=norm)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                            dropout=dropout, activation=activation, norm=norm)

    def forward(self, inputs):
        return self.conv2(self.conv1(inputs))


class DenseBlock2D(nn.Module):
    """
    2D dense convolutional block (derived from DenseNet)

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param num_layers: number of dense layers
    :param k: how many filters to add each layer
    :param bn_size: multiplicative factor for number of bottle neck layers
    :param optional kernel_size: kernel size of the convolutions
    :param optional padding: padding used for border cases ("SAME", "VALID" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, in_channels, out_channels, num_layers, k, bn_size, kernel_size=3, padding='SAME', bias=False,
                 dropout=0.0, activation=None, norm=None):
        super().__init__()

        self.num_layers = num_layers

        # dense layers
        for i in range(num_layers):
            layer = DenseLayer2D(in_channels + i * k, k, bn_size, kernel_size=kernel_size, padding=padding, bias=bias,
                                 dropout=dropout, activation=activation, norm=norm)
            self.add_module('dense%d' % (i + 1), layer)

        # transition layer
        self.transition = Conv2D(in_channels + num_layers * k, out_channels, kernel_size=kernel_size, padding=padding,
                                 dropout=dropout, activation=activation, norm=norm)

    def forward(self, x):

        # dense layers
        features = [x]
        for i in range(self.num_layers):
            dense_layer = getattr(self, 'dense%d' % (i + 1))
            new_features = dense_layer(features)
            features.append(new_features)
        x = torch.cat(features, 1)

        return self.transition(x)


class UNetConvBlock3D(nn.Module):
    """
    3D convolutional block of the classical U-Net (i.e. two sequential convolutional layers)

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param optional kernel_size: kernel size of the convolutions
    :param optional padding: padding used for border cases ("SAME", "VALID" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', norm='batch', dropout=0.0,
                 activation='relu'):
        super().__init__()

        self.conv1 = Conv3D(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                            dropout=dropout, activation=activation, norm=norm)
        self.conv2 = Conv3D(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                            dropout=dropout, activation=activation, norm=norm)

    def forward(self, inputs):
        return self.conv2(self.conv1(inputs))


class DenseBlock3D(nn.Module):
    """
    3D dense convolutional block (derived from DenseNet)

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param num_layers: number of dense layers
    :param k: how many filters to add each layer
    :param bn_size: multiplicative factor for number of bottle neck layers
    :param optional kernel_size: kernel size of the convolutions
    :param optional padding: padding used for border cases ("SAME", "VALID" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, in_channels, out_channels, num_layers, k, bn_size, kernel_size=3, padding='SAME', bias=False,
                 dropout=0.0, activation=None, norm=None):
        super().__init__()

        self.num_layers = num_layers

        # dense layers
        for i in range(num_layers):
            layer = DenseLayer3D(in_channels + i * k, k, bn_size, kernel_size=kernel_size, padding=padding, bias=bias,
                                 dropout=dropout, activation=activation, norm=norm)
            self.add_module('dense%d' % (i + 1), layer)

        # transition layer
        self.transition = Conv3D(in_channels + num_layers * k, out_channels, kernel_size=kernel_size, padding=padding,
                                 dropout=dropout, activation=activation, norm=norm)

    def forward(self, x):

        # dense layers
        features = [x]
        for i in range(self.num_layers):
            dense_layer = getattr(self, 'dense%d' % (i + 1))
            new_features = dense_layer(features)
            features.append(new_features)
        x = torch.cat(features, 1)

        return self.transition(x)


class UNetUpSamplingBlock2D(nn.Module):
    """
    2D upsampling block of the classical U-Net

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param optional deconv: us deconvolution or upsampling layers
    :param optional bias: use bias term or not
    """

    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super().__init__()

        if deconv:  # use transposed convolution
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        else:  # use bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, *arg):
        if len(arg) == 2:
            return self.forward_concat(arg[0], arg[1])
        else:
            return self.forward_standard(arg[0])

    def forward_concat(self, inputs1, inputs2):

        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):

        return self.up(inputs)


class UNetUpSamplingBlock3D(nn.Module):
    """
    3D upsampling block of the classical U-Net

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param optional deconv: us deconvolution or upsampling layers
    :param optional bias: use bias term or not
    """

    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super().__init__()

        if deconv:  # use transposed convolution
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        else:  # use bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, *arg):
        if len(arg) == 2:
            return self.forward_concat(arg[0], arg[1])
        else:
            return self.forward_standard(arg[0])

    def forward_concat(self, inputs1, inputs2):

        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):

        return self.up(inputs)

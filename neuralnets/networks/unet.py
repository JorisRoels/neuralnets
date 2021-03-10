from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import iou

from neuralnets.networks.blocks import *
from neuralnets.util.torch.metrics import iou
from neuralnets.util.tools import *
from neuralnets.util.augmentation import *
from neuralnets.util.losses import CrossEntropyLoss


class UNetEncoder(nn.Module):
    """
    U-Net encoder base class

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional dense_blocks: specify use of dense blocks
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu',
                 dense_blocks=False):
        super().__init__()

        self.features = nn.Sequential()
        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation
        self.dense_blocks = dense_blocks


class UNetDecoder(nn.Module):
    """
    U-Net decoder base class

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional residual_connections: use residual connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', dropout=0.0, activation='relu'):
        super().__init__()

        self.features = nn.Sequential()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.skip_connections = skip_connections
        self.residual_connections = residual_connections
        self.norm = norm
        self.dropout = dropout
        self.activation = activation


class UNetEncoder2D(UNetEncoder):
    """
    2D U-Net encoder

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            # convolutional block
            conv_block = UNetConvBlock2D(in_features, out_features, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)


            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = UNetConvBlock2D(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps, norm=norm,
                                           dropout=dropout, activation=activation)

    def forward(self, inputs):
        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs


class UNetEncoder3D(UNetEncoder):
    """
    3D U-Net encoder

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            # convolutional block
            conv_block = UNetConvBlock3D(in_features, out_features, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = UNetConvBlock3D(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps, norm=norm,
                                           dropout=dropout, activation=activation)

    def forward(self, inputs):
        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs


class UNetDecoder2D(UNetDecoder):
    """
    2D U-Net decoder

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional residual_connections: use residual connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', dropout=0.0, activation='relu'):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                         skip_connections=skip_connections, residual_connections=residual_connections, norm=norm,
                         dropout=dropout, activation=activation)

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock2D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections and not residual_connections:
                conv_block = UNetConvBlock2D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            else:
                conv_block = UNetConvBlock2D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv2d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        outputs = inputs
        for i in range(self.levels):
            if self.skip_connections:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[self.levels - i - 1],
                                                                       outputs)  # also deals with concat
            elif self.residual_connections:
                outputs = encoder_outputs[self.levels - i - 1] + \
                          getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # residual connection
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class UNetDecoder3D(UNetDecoder):
    """
    3D U-Net decoder

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional residual_connections: use residual connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', dropout=0.0, activation='relu'):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                         skip_connections=skip_connections, residual_connections=residual_connections, norm=norm,
                         dropout=dropout, activation=activation)

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock3D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections and not residual_connections:
                conv_block = UNetConvBlock3D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            else:
                conv_block = UNetConvBlock3D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv3d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        outputs = inputs
        for i in range(self.levels):
            if self.skip_connections:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[self.levels - i - 1],
                                                                       outputs)  # also deals with concat
            elif self.residual_connections:
                outputs = encoder_outputs[self.levels - i - 1] + \
                          getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # residual connection
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class DenseUNetEncoder2D(UNetEncoder):
    """
    2D Dense U-Net encoder

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional num_layers: number of dense layers
    :param optional k: how many filters to add each layer
    :param optional bn_size: multiplicative factor for number of bottle neck layers
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu',
                 num_layers=4, k=16, bn_size=2):
        super().__init__(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        self.num_layers = num_layers
        self.k = k
        self.bn_size = bn_size

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            # dense convolutional block
            conv_block = DenseBlock2D(in_features, out_features, num_layers, k, bn_size, norm=norm, dropout=dropout,
                                      activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = DenseBlock2D(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps, num_layers, k,
                                        bn_size, norm=norm, dropout=dropout, activation=activation)

    def forward(self, inputs):
        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs


class DenseUNetEncoder3D(UNetEncoder):
    """
    3D Dense U-Net encoder

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional num_layers: number of dense layers
    :param optional k: how many filters to add each layer
    :param optional bn_size: multiplicative factor for number of bottle neck layers
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu',
                 num_layers=4, k=16, bn_size=2):
        super().__init__(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        self.num_layers = num_layers
        self.k = k
        self.bn_size = bn_size

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            # dense convolutional block
            conv_block = DenseBlock3D(in_features, out_features, num_layers, k, bn_size, norm=norm, dropout=dropout,
                                      activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = DenseBlock3D(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps, num_layers, k,
                                        bn_size, norm=norm, dropout=dropout, activation=activation)

    def forward(self, inputs):
        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs


class DenseUNetDecoder2D(UNetDecoder):
    """
    2D Dense U-Net decoder

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional residual_connections: use residual connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional num_layers: number of dense layers
    :param optional k: how many filters to add each layer
    :param optional bn_size: multiplicative factor for number of bottle neck layers
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', dropout=0.0, activation='relu', num_layers=4, k=16, bn_size=2):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                         skip_connections=skip_connections, residual_connections=residual_connections, norm=norm,
                         dropout=dropout, activation=activation)

        self.num_layers = num_layers
        self.k = k
        self.bn_size = bn_size

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock2D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections and not residual_connections:
                conv_block = DenseBlock2D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                          num_layers, k, bn_size, norm=norm, dropout=dropout, activation=activation)
            else:
                conv_block = DenseBlock2D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                          num_layers, k, bn_size, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv2d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        outputs = inputs
        for i in range(self.levels):
            if self.skip_connections:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[self.levels - i - 1],
                                                                       outputs)  # also deals with concat
            elif self.residual_connections:
                outputs = encoder_outputs[self.levels - i - 1] + \
                          getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # residual connection
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class DenseUNetDecoder3D(UNetDecoder):
    """
    3D Dense U-Net decoder

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional residual_connections: use residual connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional num_layers: number of dense layers
    :param optional k: how many filters to add each layer
    :param optional bn_size: multiplicative factor for number of bottle neck layers
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', dropout=0.0, activation='relu', num_layers=4, k=16, bn_size=2):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                         skip_connections=skip_connections, residual_connections=residual_connections, norm=norm,
                         dropout=dropout, activation=activation)

        self.num_layers = num_layers
        self.k = k
        self.bn_size = bn_size

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock3D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections and not residual_connections:
                conv_block = DenseBlock3D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                          num_layers, k, bn_size, norm=norm, dropout=dropout, activation=activation)
            else:
                conv_block = DenseBlock3D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                          num_layers, k, bn_size, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv3d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        outputs = inputs
        for i in range(self.levels):
            if self.skip_connections:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[self.levels - i - 1],
                                                                       outputs)  # also deals with concat
            elif self.residual_connections:
                outputs = encoder_outputs[self.levels - i - 1] + \
                          getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # residual connection
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class UNet(pl.LightningModule):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, loss_fn=CrossEntropyLoss(), lr=1e-3):
        super().__init__()

        # parameters
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.c = in_channels // 2
        self.coi = coi
        self.out_channels = len(coi)
        self.feature_maps = feature_maps
        self.levels = levels
        self.skip_connections = skip_connections
        self.residual_connections = residual_connections
        self.norm = norm
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        self.activation = activation
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x):

        # contractive path
        encoder_outputs, final_output = self.encoder(x)

        # expansive path
        decoder_outputs, outputs = self.decoder(final_output, encoder_outputs)

        return outputs

    def training_step(self, batch, batch_idx):

        # transfer to suitable device and get labels
        x, y = batch

        # forward prop
        y_pred = self(x)

        # compute loss
        loss = self.loss_fn(y_pred, y[:, 0, ...])

        # compute iou
        mIoU = 0
        y_pred = torch.softmax(y_pred, dim=1)
        for c in range(y_pred.size(1)):
            mIoU += iou(y_pred[:, c:c+1, ...], y == c, w=(y != 255))
        mIoU /= y_pred.size(1)
        self.log('train/mIoU', mIoU, prog_bar=True)
        self.log('train/loss', loss)

        # log images
        if batch_idx == 0:
            self._log_predictions(x, y, y_pred, prefix='train')

        return loss

    def validation_step(self, batch, batch_idx):

        # transfer to suitable device and get labels
        x, y = batch

        # forward prop
        y_pred = self(x)

        # compute loss
        loss = self.loss_fn(y_pred, y[:, 0, ...])

        # compute iou
        mIoU = 0
        y_pred = torch.softmax(y_pred, dim=1)
        for c in range(y_pred.size(1)):
            mIoU += iou(y_pred[:, c:c+1, ...], y == c, w=(y != 255))
        mIoU /= y_pred.size(1)
        self.log('val/mIoU', mIoU, prog_bar=True)
        self.log('val/loss', loss)

        # log images
        if batch_idx == 0:
            self._log_predictions(x, y, y_pred, prefix='val')

        return loss

    def test_step(self, batch, batch_idx):

        # transfer to suitable device and get labels
        x, y = batch

        # forward prop
        y_pred = self(x)

        # compute loss
        loss = self.loss_fn(y_pred, y[:, 0, ...])

        # compute iou
        mIoU = 0
        y_pred = torch.softmax(y_pred, dim=1)
        for c in range(y_pred.size(1)):
            mIoU += iou(y_pred[:, c:c+1, ...], y == c, w=(y != 255))
        mIoU /= y_pred.size(1)
        self.log('test/mIoU', mIoU, prog_bar=True)
        self.log('test/loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'max')
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'val/mIoU'}

    def on_epoch_start(self):
        set_seed(rnd.randint(100000))

    def _log_predictions(self, x, y, y_pred, prefix='train'):
        tensorboard = self.logger.experiment
        if x.ndim == 4:  # 2D data
            tensorboard.add_image(prefix + '/input', x[0, 0:1], global_step=self.current_epoch)
            tensorboard.add_image(prefix + '/target', y[0, 0:1], global_step=self.current_epoch)
            tensorboard.add_image(prefix + '/pred', y_pred[0, 1:2], global_step=self.current_epoch)
        else:  # 3D data
            s = x.size(2) // 2
            tensorboard.add_image(prefix + '/input', x[0, 0, s:s+1], global_step=self.current_epoch)
            tensorboard.add_image(prefix + '/target', y[0, 0, s:s+1], global_step=self.current_epoch)
            tensorboard.add_image(prefix + '/pred', y_pred[0, 1, s:s+1], global_step=self.current_epoch)


class UNet2D(UNet):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, loss_fn=CrossEntropyLoss(), lr=1e-3):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr)

        # contractive path
        self.encoder = UNetEncoder2D(in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                     dropout=dropout_enc, activation=activation)
        # expansive path
        self.decoder = UNetDecoder2D(self.out_channels, feature_maps=feature_maps, levels=levels,
                                     skip_connections=skip_connections, residual_connections=residual_connections,
                                     norm=norm, dropout=dropout_dec, activation=activation)


class UNet3D(UNet):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, loss_fn=CrossEntropyLoss(), lr=1e-3):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr)

        # contractive path
        self.encoder = UNetEncoder3D(in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                     dropout=dropout_enc, activation=activation)
        # expansive path
        self.decoder = UNetDecoder3D(self.out_channels, feature_maps=feature_maps, levels=levels,
                                     skip_connections=skip_connections, residual_connections=residual_connections,
                                     norm=norm, dropout=dropout_dec, activation=activation)


class DenseUNet2D(UNet):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, num_layers=4, k=16, bn_size=2, loss_fn=CrossEntropyLoss(), lr=1e-3):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr)

        # parameters
        self.num_layers = num_layers
        self.k = k
        self.bn_size = bn_size

        # contractive path
        self.encoder = DenseUNetEncoder2D(in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                          dropout=dropout_enc, activation=activation, num_layers=num_layers, k=k,
                                          bn_size=bn_size)
        # expansive path
        self.decoder = DenseUNetDecoder2D(self.out_channels, feature_maps=feature_maps, levels=levels,
                                          skip_connections=skip_connections, residual_connections=residual_connections,
                                          norm=norm, dropout=dropout_dec, activation=activation, num_layers=num_layers,
                                          k=k, bn_size=bn_size)


class DenseUNet3D(UNet):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, num_layers=4, k=16, bn_size=2, loss_fn=CrossEntropyLoss(), lr=1e-3):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr)

        # parameters
        self.num_layers = num_layers
        self.k = k
        self.bn_size = bn_size

        # contractive path
        self.encoder = DenseUNetEncoder3D(in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                          dropout=dropout_enc, activation=activation, num_layers=num_layers, k=k,
                                          bn_size=bn_size)
        # expansive path
        self.decoder = DenseUNetDecoder3D(self.out_channels, feature_maps=feature_maps, levels=levels,
                                          skip_connections=skip_connections, residual_connections=residual_connections,
                                          norm=norm, dropout=dropout_dec, activation=activation, num_layers=num_layers,
                                          k=k, bn_size=bn_size)

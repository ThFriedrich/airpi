

import tensorflow as tf
from ap_utils.functions import tf_pi
from tensorflow import keras as tfk
from cvnn.layers.convolutional import ComplexConv2D, ComplexConv2DTranspose
from cvnn.layers.core import ComplexDropout, ComplexBatchNormalization
from cvnn import activations as ComplexActivation


kl = tfk.layers     


class Cast_Input(tfk.layers.Layer):
    """Cast Input Layer (for complex inputs)"""

    def __init__(self):
        super(Cast_Input, self).__init__()


    def call(self, x, training=False):
        probe = tf.cast(x[..., 9:], tf.complex64)
        probe = (probe[..., 0]*tf.exp(1j*probe[..., 1]))[..., tf.newaxis]

        x = tf.cast(x[..., :9], tf.complex64)
        x = tf.concat((x, probe), -1)

        return x, probe

class Convolution_Block_Complex(tfk.layers.Layer):
    """Layer implementing a sequence of complex 2D-Convolution, normalization and activation."""

    def __init__(self, filters=None, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=2, normalization=1, dropout=None, transpose=False):
        super(Convolution_Block_Complex, self).__init__()
        self.filters = int(filters)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.transpose = transpose
        if dropout is not None and dropout > 0.0:
            self.dropout_layer = ComplexDropout(rate=dropout) 
        else:
            self.dropout_layer = None

        if self.activation == 0:
            self.activation_layer = ComplexActivation.linear
        elif self.activation == 1:
            self.activation_layer = ComplexActivation.zrelu
        elif self.activation == 2:
            self.activation_layer = ComplexActivation.complex_cardioid
        elif self.activation == 3:
            self.activation_layer = ComplexActivation.crelu
        elif self.activation == 4:
            self.activation_layer = ComplexActivation.modrelu
        else:
            self.activation_layer = ComplexActivation.linear

        if self.normalization == 0:
            self.normalization_layer = None
        elif self.normalization == 1:
            self.normalization_layer = ComplexBatchNormalization()
        else:
            self.normalization_layer = None

    def build(self, input_shape):
        if self.filters is None:
            self.filters = input_shape[-1]
        if self.transpose:
            self.conv2d = ComplexConv2DTranspose(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                        activation=None)
        else:
            self.conv2d = ComplexConv2D(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                        activation=None)

    def call(self, x):
        x = self.conv2d(x)
        if self.normalization_layer is not None:
            x = self.normalization_layer(x)
        if self.activation_layer is not None:
            x = self.activation_layer(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return x


class Conv_Stack_Complex(tfk.layers.Layer):
    """Layer implementing a sequence of complex Convolution Blocks (2D-Convolution, normalization and activation)."""

    def __init__(self, n_blocks=3, filters=None, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=2, normalization=1, dropout=None):
        super(Conv_Stack_Complex, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.n_blocks = n_blocks
        self.dropout = dropout

    def build(self, input_shape):
        if self.filters is None:
            self.filters = input_shape[-1]
        self.conv_blocks = []
        for _ in range(self.n_blocks):
            self.conv_blocks.append(Convolution_Block_Complex(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                                      activation=self.activation, normalization=self.normalization, dropout=self.dropout))

    def call(self, x, training=False):
        for block in self.conv_blocks:
            x = block(x)
        return x


class Contraction_Block_Complex(tfk.layers.Layer):
    """Block combining a complex Convolution Stack and a `n` strided convolution layer, raising the number of filters by a factor of `n`."""

    def __init__(self, n_blocks=3, kernel_size=[3, 3], padding='same', activation=2, normalization=1, contraction_fct=2, dropout=None):
        super(Contraction_Block_Complex, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.contraction_fct = int(contraction_fct)
        self.strides = [self.contraction_fct, self.contraction_fct]
        self.dropout = dropout

    def build(self, input_shape):
        nf = int(input_shape[-1]*self.contraction_fct)
        self.conv_stack = Conv_Stack_Complex(n_blocks=self.n_blocks, kernel_size=self.kernel_size, strides=[1, 1], padding=self.padding,
                                     activation=self.activation, normalization=self.normalization, dropout=self.dropout)
        self.contraction = Convolution_Block_Complex(filters=nf,  kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,                                             
                                             activation=self.activation, normalization=self.normalization, dropout=self.dropout)

    def call(self, x):
        c = self.conv_stack(x)
        x = self.contraction(c)
        return x, c


class Expansion_Block_Complex(tfk.layers.Layer):
    """Block combining a complex Convolution Stack and a `n` strided transpose convolution layer, reducing the number of filters by a factor of `n`."""

    def __init__(self, n_blocks=3, kernel_size=[3, 3], padding='same', activation=2, normalization=1, expansion_fct=2, dropout=None):
        super(Expansion_Block_Complex, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.expansion_fct = int(expansion_fct)
        self.strides = [self.expansion_fct, self.expansion_fct]
        self.dropout = dropout

    def build(self, input_shape):
        nf = int(input_shape[-1]//self.expansion_fct)
        self.expansion = Convolution_Block_Complex(filters=nf,  kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                           activation=self.activation, normalization=self.normalization, transpose=True, dropout=self.dropout)
        self.conv_stack = Conv_Stack_Complex(filters=nf,n_blocks=self.n_blocks, kernel_size=self.kernel_size, strides=[1, 1], padding=self.padding,
                                     activation=self.activation, normalization=self.normalization, dropout=self.dropout)
        self.concatenation = tfk.layers.Concatenate()

    def call(self, x, c):
        x = self.expansion(x)
        x = self.concatenation([x, c])
        x = self.conv_stack(x)
        return x


import tensorflow as tf
from ap_architectures.utils import tf_pi, tf_sin_cos2rad, floatx
from tensorflow import keras as tfk
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization

kl = tfk.layers


class amplitude_output(tfk.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, b_skip=True):
        super(amplitude_output, self).__init__()
        self.scale =  tf.Variable(initial_value=0.015625, dtype=tf.float32, trainable=False)
        self.b_skip = b_skip
        self.conv = kl.Conv2D(1, kernel_size=[3, 3], padding='same',
                              strides=[1, 1], kernel_regularizer=None, activity_regularizer=None,
                              activation='linear', dtype=tf.float32)

    def df_bf_ratio_loss(self, x0, x_i):
        b_msk = tf.cast(x0[..., 9], tf.bool)
        bf_t = tf.math.reduce_mean(tf.where(b_msk, x0[..., 4], 0), axis=[1, 2])
        df_t = tf.math.reduce_mean(
            tf.where(~b_msk, x0[..., 4], 0), axis=[1, 2])
        ratio_t = tf.math.maximum(tf.math.divide_no_nan(df_t, bf_t), 1e-7)
        bf_p = tf.math.reduce_mean(
            tf.where(b_msk, tf.squeeze(x_i), 0), axis=[1, 2])
        df_p = tf.math.reduce_mean(
            tf.where(~b_msk, tf.squeeze(x_i), 0), axis=[1, 2])
        ratio_p = tf.math.maximum(tf.math.divide_no_nan(df_p, bf_p), 1e-7)
        ls = tf.math.reduce_mean(tf.math.abs(ratio_p - ratio_t))/10

        self.add_loss(ls)
        self.add_metric(ls, name='df_ratio_k2')

    def call(self, x0, x, training=False):

        x = self.conv(x)
        if self.b_skip:
            dose = tf.reduce_sum(x0[..., 4])
            x_add = (x0[..., 4, tf.newaxis]/dose)**0.05
            x = kl.add([x, x_add])

        if training:
            x = x * self.scale
        else:
            x = tf.maximum(0.0, x) * self.scale
        
        ls =  tf.math.reduce_mean(tf.maximum(tf.math.reduce_sum(x**10, axis=[1, 2])-1,0))
        
        self.add_loss(ls)
        self.add_metric(ls, name='int_constraint')
        self.add_metric(self.scale, name='amp_scale')

        # self.df_bf_ratio_loss(x0, x)

        return x


class phase_output(tfk.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, n=1):
        super(phase_output, self).__init__()
        self.scale =  tf.Variable(initial_value=0.2, dtype=tf.float32, trainable=False, constraint=lambda x: tf.clip_by_value(x, 1e-3, 1))
        self.n = n
        self.conv = kl.Conv2D(self.n, kernel_size=[3, 3], padding='same',
                              strides=[1, 1], kernel_regularizer=None, activity_regularizer=None,
                              activation='linear', dtype=tf.float32)

    def decompose(self, x):
        if self.n > 1:
            x_sin, x_cos = tf.split(x)
            x_sin = tf.math.sin(x_sin)
            x_cos = tf.math.cos(x_cos)
            ls = tf.norm(tf.stack([x_sin, x_cos]),
                         ord="euclidean", axis=0) - 1.0
            self.add_loss(ls)
            self.add_metric(ls, name='sin_cos_ecn')
        else:
            x_sin = tf.sin(x * self.scale)
            x_cos = tf.cos(x * self.scale)
        return x_sin, x_cos

    def add_constraint(self, x_p):
        xu = tf.where(x_p > tf_pi, x_p , 0)
        xl = tf.where(x_p < -tf_pi, x_p , 0)
        ls = tf.math.reduce_sum(tf.math.abs(xu)) + tf.math.reduce_sum(tf.math.abs(xl))
        self.add_loss(ls)
        self.add_metric(ls, name='phase_constraint')
        self.add_metric(self.scale, name='phase_scale')

    def call(self, x):
        x = self.conv(x) * self.scale
        self.add_constraint(x)
        # x = tf.math.tanh(x)*tf_pi
        # x *= tf_pi
        # x_sin, x_cos = self.decompose(x)

        return x

class Deploy_Output(tfk.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self):
        super(Deploy_Output, self).__init__()

    def cast(self, x):
        x_a = tf.cast(x[...,0]**5, tf.complex64)
        bl = tf.reduce_mean(x[:,31:33,31:33,1])-tf.expand_dims(tf.reduce_mean(x[:,31:33,31:33,1],axis=[1,2]),-1)[...,tf.newaxis]
        # bl = tf.reduce_mean(x[:,32,32,1],keepdims=True)-tf.expand_dims(x[:,32,32,1],-1)[...,tf.newaxis]
        x_p = tf.cast(x[...,1]-bl, tf.complex64)
        x_o = tf.squeeze((x_a * tf.exp(1j*x_p)))
        return x_o
    
    def call(self, x):
        x = self.cast(x)
        return x

class Standardization_Layer(tfk.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self):
        super(Standardization_Layer, self).__init__()

    def flatten(self, x: tf.Tensor) -> tf.Tensor:
        return x**0.1

    def standardize(self, x: tf.Tensor) -> tf.Tensor:
        x_mean, x_var = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_std = tf.maximum(tf.sqrt(x_var), 1e-8)
        x = tf.divide(tf.subtract(x, x_mean), x_std)

        return x

    def x_normalise(self, x: tf.Tensor) -> tf.Tensor:
        # xf = self.flatten(xf)
        xf = self.standardize(self.flatten(x[..., 0:9]))
        if x.shape[-1] > 9:
            msk_a = self.standardize(x[..., 9, tf.newaxis])
            xf = tf.concat((xf, msk_a), -1)
        if x.shape[-1] > 10:
            msk_p = self.standardize(x[...,10, tf.newaxis])
            xf = tf.concat((xf, msk_p), -1)
        return xf

    def call(self, x):
        return self.x_normalise(x)


class Block_Normalization(tfk.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, momentum=0.9, epsilon=1e-8):
        super(Block_Normalization, self).__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = self.add_weight(
            name='gamma', shape=(1,), initializer='ones')
        self.beta = self.add_weight(
            name='beta', shape=(1,), initializer='zeros')
        self.moving_mean = tf.Variable(
            0.0, name='moving_mean', trainable=False)
        self.moving_var = tf.Variable(1.0, name='moving_var', trainable=False)

    def normalize(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        if training:
            x_mean, x_var = tf.nn.moments(x, axes=[0, 1, 2, 3], keepdims=False)
            x_std = tf.sqrt(x_var + self.epsilon)
            self.moving_mean.assign(
                self.moving_mean * self.momentum + x_mean * (1 - self.momentum))
            self.moving_var.assign(
                self.moving_var * self.momentum + x_var * (1 - self.momentum))
            return self.gamma * (x - x_mean) / x_std + self.beta
        else:
            return self.gamma * (x - self.moving_mean) / tf.sqrt(self.moving_var + self.epsilon) + self.beta

    def call(self, x, training=False):
        return self.normalize(x, training)


class Grouped_Batch_Normalization(tfk.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, groups=2):
        super(Grouped_Batch_Normalization, self).__init__()
        self.groups = groups
        self.batch_norm = []
        for _ in range(groups):
            self.batch_norm.append(Block_Normalization())

    def normalise(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        grps = tf.split(x, self.groups, -1)
        for ig, g in enumerate(grps):
            grps[ig] = self.batch_norm[ig](g, training)
        return tf.concat(grps, -1)

    def call(self, x, training=None):
        return self.normalise(x, training)


class Convolution_Block(tfk.layers.Layer):
    """Layer implementing a sequence of 2D-Convolution, normalization and activation."""

    def __init__(self, filters=None, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=5, normalization=5, dropout=None, initializer='glorot_uniform', kernel_regularizer=None, activity_regularizer=None, transpose=False):
        super(Convolution_Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer
        self.transpose = transpose

        self.dropout_layer = kl.Dropout(self.dropout)

        if self.activation == 0:
            self.activation_layer = None
        elif self.activation == 1:
            self.activation_layer = kl.LeakyReLU()
        elif self.activation == 2:
            self.activation_layer = kl.ELU()
        elif self.activation == 3:
            self.activation_layer = tfk.activations.swish
        elif self.activation == 5:
            self.activation_layer = kl.ReLU()
        else:
            self.activation_layer = kl.Activation('linear')

        if self.normalization == 0:
            self.normalization_layer = None
        elif self.normalization == 1:
            self.normalization_layer = InstanceNormalization()
        elif self.normalization == 2:
            self.normalization_layer = kl.LayerNormalization()
        elif self.normalization == 3:
            self.normalization_layer = Grouped_Batch_Normalization()
        elif self.normalization == 4:
            self.normalization_layer = GroupNormalization(groups=2)
        elif self.normalization == 5:
            self.normalization_layer = kl.BatchNormalization()
        else:
            self.normalization_layer = None

    def build(self, input_shape):
        if self.filters is None:
            self.filters = input_shape[-1]
        if self.transpose:
            self.conv2d = tfk.layers.Conv2DTranspose(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                                     kernel_regularizer=self.kernel_regularizer, activity_regularizer=self.activity_regularizer, kernel_initializer=self.initializer,
                                                     activation=None, dtype=tf.float32)
        else:
            self.conv2d = kl.Conv2D(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                    kernel_regularizer=self.kernel_regularizer, activity_regularizer=self.activity_regularizer, kernel_initializer=self.initializer,
                                    activation=None, dtype=tf.float32)

    def call(self, x, training=False):
        x = self.conv2d(x)
        if self.normalization_layer is not None:
            x = self.normalization_layer(x, training)
        if self.activation_layer is not None:
            x = self.activation_layer(x)
        if self.dropout is not None:
            x = self.dropout_layer(x)
        return x


class Conv_Stack(tfk.layers.Layer):
    """Layer implementing a sequence of 2D-Convolution, normalization and activation."""

    def __init__(self, n_blocks=3, filters=None, kernel_size=[3,3], strides=[1,1], padding='same', activation=5, normalization=5, dropout=None, initializer='glorot_uniform', kernel_regularizer=None, activity_regularizer=None):
        super(Conv_Stack, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer
        self.n_blocks = n_blocks

    def build(self, input_shape):
        if self.filters is None:
            self.filters = input_shape[-1]
        self.conv_blocks = []
        for _ in range(self.n_blocks):
            self.conv_blocks.append(Convolution_Block(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                            kernel_regularizer=self.kernel_regularizer, activity_regularizer=self.activity_regularizer, initializer=self.initializer,
                                            activation=self.activation, normalization=self.normalization, dropout=self.dropout))

    def call(self, x, training=False):
        for block in self.conv_blocks:
            x = block(x, training)
        return x


class Contraction_Block(tfk.layers.Layer):
    """Layer implementing a sequence of 2D-Convolution, normalization and activation."""

    def __init__(self, n_blocks=3, kernel_size=[3, 3], padding='same', activation=5, normalization=5, dropout=None, initializer='glorot_uniform', kernel_regularizer=None, activity_regularizer=None, contraction_fct=2):
        super(Contraction_Block, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.initializer = initializer
        self.contraction_fct = int(contraction_fct)
        self.strides = [self.contraction_fct, self.contraction_fct]
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer

    def build(self, input_shape):
        nf = int(input_shape[-1]*self.contraction_fct)
        self.conv_stack = Conv_Stack(n_blocks=self.n_blocks, kernel_size=self.kernel_size, strides=[1,1], padding=self.padding,
                                     kernel_regularizer=self.kernel_regularizer, activity_regularizer=self.activity_regularizer, initializer=self.initializer,
                                     activation=self.activation, normalization=self.normalization, dropout=self.dropout)
        self.contraction = Convolution_Block(filters=nf,  kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                             kernel_regularizer=self.kernel_regularizer, activity_regularizer=self.activity_regularizer, initializer=self.initializer,
                                             activation=self.activation, normalization=self.normalization, dropout=self.dropout)

    def call(self, x, training=False):
        c = self.conv_stack(x, training)
        x = self.contraction(c, training)
        return x, c


class Expansion_Block(tfk.layers.Layer):
    """Layer implementing a sequence of 2D-Convolution, normalization and activation."""

    def __init__(self, n_blocks=3, kernel_size=[3, 3], padding='same', activation=5, normalization=5, dropout=None, initializer='glorot_uniform', kernel_regularizer=None, activity_regularizer=None, expansion_fct=2):
        super(Expansion_Block, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.initializer = initializer
        self.expansion_fct = int(expansion_fct)
        self.strides = [self.expansion_fct, self.expansion_fct]
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer

    def build(self, input_shape):
        nf = int(input_shape[-1]//self.expansion_fct)
        self.expansion = Convolution_Block(filters=nf,  kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                             kernel_regularizer=self.kernel_regularizer, activity_regularizer=self.activity_regularizer, initializer=self.initializer,
                                             activation=self.activation,normalization=self.normalization,transpose=True)
        self.conv_stack = Conv_Stack(n_blocks=self.n_blocks, kernel_size=self.kernel_size, strides=[1,1], padding=self.padding,
                                     kernel_regularizer=self.kernel_regularizer, activity_regularizer=self.activity_regularizer, initializer=self.initializer,
                                     activation=self.activation,normalization=self.normalization)
        self.concatenation = tfk.layers.Concatenate()

    def call(self, x, c, training=False):
        x = self.expansion(x, training)
        x = self.concatenation([x, c])
        x = self.conv_stack(x, training)
        return x
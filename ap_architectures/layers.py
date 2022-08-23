
import tensorflow as tf
from ap_utils.functions import tf_fft2d, tf_pi
from tensorflow.keras import layers as kl, activations as ka
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization


class Amplitude_Output(kl.Layer):
    """Amplitude Output Layer"""

    def __init__(self, b_skip=False, b_cat=False):
        super(Amplitude_Output, self).__init__()
        self.scale =  tf.Variable(initial_value=0.00277, dtype=tf.float32, trainable=False)
        self.b_skip = b_skip
        self.b_cat = b_cat
        self.conv = kl.Conv2D(1, kernel_size=[3, 3], padding='same',
                              strides=[1, 1], kernel_regularizer=None, activity_regularizer=None,
                              activation='linear', dtype=tf.float32)

    def call(self, x0, x, training=False):

        if self.b_cat:
            beam = (x0[...,9, tf.newaxis])**0.2
            x = tf.concat([x, beam], axis=-1)

        x = self.conv(x) * self.scale
        

        if self.b_skip:
            x_add = (x0[...,9, tf.newaxis])**0.2
            x = kl.add([x, x_add])

            
        if training:
            int = tf.math.reduce_sum(x**10, axis=[1, 2])
            ls =  tf.math.reduce_mean(tf.maximum(int-1,0))
            ls +=  tf.math.reduce_mean(tf.maximum(0.3-int,0))
            self.add_loss(ls)
            self.add_metric(ls, name='int_regularization')
        if self.scale.trainable:
            self.add_metric(self.scale, name='amp_scale')
          
        return x
        

class Phase_Output(kl.Layer):
    """Phase Output Layer"""

    def __init__(self, b_skip=False, b_cat=False):
        super(Phase_Output, self).__init__()
        self.scale =  tf.Variable(initial_value=1e-6, dtype=tf.float32, trainable=True)
        self.b_skip = b_skip
        self.b_cat = b_cat
        self.conv_s = kl.Conv2D(1, kernel_size=[3, 3], padding='same',
                              strides=[1, 1], kernel_regularizer=None, activity_regularizer=None,
                              activation='linear', dtype=tf.float32)

    def add_constraint(self, x_p, x0, x_s=None, x_c=None):
        xu = tf.where(x_p > tf_pi, x_p , 0)
        xl = tf.where(x_p < -tf_pi, x_p , 0)
        ls = tf.math.reduce_sum(tf.math.abs(xu)) + tf.math.reduce_sum(tf.math.abs(xl), axis=[1, 2])
        ls = tf.math.reduce_mean(ls)
        self.add_loss(ls)
        self.add_metric(ls, name='phase_regularization')

    def call(self, x0, x, training=False):

        if self.b_cat:
            x_add = x0[...,10, tf.newaxis]
            x = tf.concat([x, x_add], axis=-1)

        x = self.conv_s(x) * self.scale

        if self.b_skip:
            x_add = x0[...,10, tf.newaxis]
            x = kl.add([x, x_add])

        self.add_constraint(x, x0)

        if self.scale.trainable:
            self.add_metric(self.scale, name='phase_scale')

        return x

class Deploy_Output(kl.Layer):
    """Deploy output layer (only used for deployment)"""

    def __init__(self):
        super(Deploy_Output, self).__init__()
    
    def cast_complex(self, amp, phase):
        return tf.cast(amp, tf.complex64) * tf.math.exp(tf.cast(phase, tf.complex64) * 1j)

    def cast(self, x0, x):
        x_a = tf.cast(x[...,0]**5, tf.complex64)
        x_p = tf.cast(x[...,1], tf.complex64)
        x_o = tf.squeeze((x_a * tf.exp(1j*x_p)))
        x_i = self.cast_complex(x0[..., -2], x0[..., -1])
        obj = x_o/x_i
        obj = tf.cast(tf.math.exp(tf.cast(tf.math.angle(obj),tf.complex64)*1j),tf.complex64)
        # obj = tf_fft2d(obj)
        return obj
    
    def call(self, x0, x):
        return self.cast(x0, x)

class Standardization_Layer(kl.Layer):
    """Standardization layer. Normalizes the input tensor to have zero mean and unit variance"""

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


class Block_Normalization(kl.Layer):
    """Block normalization layer. Normalizes the input tensor to have zero mean and unit variance
    across the entire tensor (including the batch dimension)"""

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


class Grouped_Batch_Normalization(kl.Layer):
    """Grouped batch normalization layer. Normalizes the input tensor to have zero mean and unit variance
    across `n` groups (including the batch dimension)"""

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


class Convolution_Block(kl.Layer):
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

        if dropout is not None and dropout > 0.0:
            self.dropout_layer = kl.Dropout(self.dropout)
        else:
            self.dropout_layer = None

        if self.activation == 0:
            self.activation_layer = None
        elif self.activation == 1:
            self.activation_layer = kl.LeakyReLU()
        elif self.activation == 2:
            self.activation_layer = kl.ELU()
        elif self.activation == 3:
            self.activation_layer = ka.swish
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
            self.conv2d = kl.Conv2DTranspose(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
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
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return x


class Conv_Stack(kl.Layer):
    """Layer implementing a sequence of Convolution Blocks (2D-Convolution, normalization and activation)."""

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


class Contraction_Block(kl.Layer):
    """Block combining a Convolution Stack and a `n` strided convolution layer, raising the number of filters by a factor of `n`."""

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


class Expansion_Block(kl.Layer):
    """Block combining a Convolution Stack and a `n` strided transpose convolution layer, reducing the number of filters by a factor of `n`."""

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
                                             activation=self.activation,normalization=self.normalization,transpose=True, dropout=self.dropout)
        self.conv_stack = Conv_Stack(filters=nf, n_blocks=self.n_blocks, kernel_size=self.kernel_size, strides=[1,1], padding=self.padding,
                                     kernel_regularizer=self.kernel_regularizer, activity_regularizer=self.activity_regularizer, initializer=self.initializer,
                                     activation=self.activation,normalization=self.normalization, dropout=self.dropout)
        self.concatenation = kl.Concatenate()

    def call(self, x, c, training=False):
        x = self.expansion(x, training)
        x = self.concatenation([x, c])
        x = self.conv_stack(x, training)
        return x

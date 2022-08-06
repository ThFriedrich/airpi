

import tensorflow as tf
from ap_architectures.utils import tf_pi
from tensorflow import keras as tfk
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow.keras.backend as K
# from tf_complex.convolutions import ComplexConv2D, ComplexActivation
from cvnn.layers.convolutional import ComplexConv2D, ComplexConv2DTranspose
from cvnn.layers.core import ComplexDropout, ComplexBatchNormalization, ComplexInput
from cvnn import activations as ComplexActivation


kl = tfk.layers

class Amplitude_Output_Complex(tfk.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, b_skip=True):
        super(Amplitude_Output_Complex, self).__init__()
        self.scale =  tf.Variable(initial_value=0.0196, dtype=tf.float32, trainable=False)
        self.b_skip = b_skip

    def call(self, x0, x, training=False):

        x = tf.abs(x)
        if training:
            x = x * self.scale
        else:
            x = tf.maximum(0.0, x) * self.scale
        
        if training:
            ls =  tf.math.reduce_mean(tf.maximum(tf.math.reduce_sum(x**10, axis=[1, 2])-1,0))
            self.add_loss(ls)
            self.add_metric(ls, name='int_constraint')
        
        
        return x
        

class Phase_Output_Complex(tfk.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, b_skip=True):
        super(Phase_Output_Complex, self).__init__()
        self.scale =  tf.Variable(initial_value=0.2, dtype=tf.float32, trainable=True)
        self.b_skip = b_skip

    def add_constraint(self, x_p, x0):
        xu = tf.where(x_p > tf_pi, x_p , 0)
        xl = tf.where(x_p < -tf_pi, x_p , 0)
        ls = tf.math.reduce_sum(tf.math.abs(xu)) + tf.math.reduce_sum(tf.math.abs(xl))
        self.add_loss(ls)
        self.add_metric(ls, name='phase_constraint')
        self.add_metric(self.scale, name='phase_scale')


    def call(self, x0, x, training=False):
        x = tf.math.angle(x) * self.scale

        if training:
            self.add_constraint(x, x0)

        return x

class Convolution_Block_Complex(tfk.layers.Layer):
    """Layer implementing a sequence of 2D-Convolution, normalization and activation."""

    def __init__(self, filters=None, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=2, normalization=1, transpose=False):
        super(Convolution_Block_Complex, self).__init__()
        self.filters = int(filters)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.transpose = transpose

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
            # self.conv2d = kl.UpSampling2D(size=self.strides, interpolation='nearest')
            # self.conv2d = kl.Lambda(lambda x:tf.nn.depth_to_space(x,2))
        else:
            self.conv2d = ComplexConv2D(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                        activation=None)

    def call(self, x):
        x = self.conv2d(x)
        if self.normalization_layer is not None:
            x = self.normalization_layer(x)
        if self.activation_layer is not None:
            x = self.activation_layer(x)
        return x


class Conv_Stack_Complex(tfk.layers.Layer):
    """Layer implementing a sequence of 2D-Convolution, normalization and activation."""

    def __init__(self, n_blocks=3, filters=None, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=2, normalization=1):
        super(Conv_Stack_Complex, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.n_blocks = n_blocks

    def build(self, input_shape):
        if self.filters is None:
            self.filters = input_shape[-1]
        self.conv_blocks = []
        for _ in range(self.n_blocks):
            self.conv_blocks.append(Convolution_Block_Complex(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                                      activation=self.activation, normalization=self.normalization))

    def call(self, x, training=False):
        for block in self.conv_blocks:
            x = block(x)
        return x


class Contraction_Block_Complex(tfk.layers.Layer):
    """Layer implementing a sequence of 2D-Convolution, normalization and activation."""

    def __init__(self, n_blocks=3, kernel_size=[3, 3], padding='same', activation=2, normalization=1, contraction_fct=2):
        super(Contraction_Block_Complex, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.contraction_fct = int(contraction_fct)
        self.strides = [self.contraction_fct, self.contraction_fct]

    def build(self, input_shape):
        nf = int(input_shape[-1]*self.contraction_fct)
        self.conv_stack = Conv_Stack_Complex(n_blocks=self.n_blocks, kernel_size=self.kernel_size, strides=[1, 1], padding=self.padding,
                                     activation=self.activation, normalization=self.normalization)
        self.contraction = Convolution_Block_Complex(filters=nf,  kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,                                             
                                             activation=self.activation, normalization=self.normalization)

    def call(self, x):
        c = self.conv_stack(x)
        x = self.contraction(c)
        return x, c


class Expansion_Block_Complex(tfk.layers.Layer):
    """Layer implementing a sequence of 2D-Convolution, normalization and activation."""

    def __init__(self, n_blocks=3, kernel_size=[3, 3], padding='same', activation=2, normalization=1, expansion_fct=2):
        super(Expansion_Block_Complex, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.expansion_fct = int(expansion_fct)
        self.strides = [self.expansion_fct, self.expansion_fct]

    def build(self, input_shape):
        nf = int(input_shape[-1]//self.expansion_fct)
        self.expansion = Convolution_Block_Complex(filters=nf,  kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                           activation=self.activation, normalization=self.normalization, transpose=True)
        self.conv_stack = Conv_Stack_Complex(filters=nf,n_blocks=self.n_blocks, kernel_size=self.kernel_size, strides=[1, 1], padding=self.padding,
                                     activation=self.activation, normalization=self.normalization)
        self.concatenation = tfk.layers.Concatenate()

    def call(self, x, c):
        x = self.expansion(x)
        x = self.concatenation([x, c])
        x = self.conv_stack(x)
        return x

# Complex Batch Normalization Layer copied and adapted from https://github.com/ChihebTrabelsi/deep_complex_networks
#
# Authors: Chiheb Trabelsi, Olexa Bilaniuk
#          modified by: Thomas Friedrich
#
# Note: The implementation of complex Batchnorm is based on
#       the Keras implementation of batch Normalization
#       available here:
#       https://github.com/fchollet/keras/blob/master/keras/layers/normalization.py


def sqrt_init(shape, dtype=None):
    value = (1.0 / tf.math.sqrt(2.0)) * tf.ones(shape)
    return value


def sanitizedInitGet(init):
    if init in ["sqrt_init"]:
        return sqrt_init
    else:
        return initializers.get(init)


def sanitizedInitSer(init):
    if init in [sqrt_init]:
        return "sqrt_init"
    else:
        return initializers.serialize(init)


def real2complex(x):
    channel = x.shape[-1] // 2
    if x.shape.ndims == 3:
        return tf.complex(x[:, :, :channel], x[:, :, channel:])
    elif x.shape.ndims == 4:
        return tf.complex(x[:, :, :, :channel], x[:, :, :, channel:])


def complex2real(x):
    x_real = tf.math.real(x)
    x_imag = tf.math.imag(x)
    return tf.concat([x_real, x_imag], axis=-1)


def complex_standardization(input_centred, Vrr, Vii, Vri,
                            layernorm=False, axis=-1):
    ndim = K.ndim(input_centred)
    input_dim = K.shape(input_centred)[axis] // 2
    variances_broadcast = [1] * ndim
    variances_broadcast[axis] = input_dim
    if layernorm:
        variances_broadcast[0] = K.shape(input_centred)[0]

    # We require the covariance matrix's inverse square root. That first requires
    # square rooting, followed by inversion (I do this in that order because during
    # the computation of square root we compute the determinant we'll need for
    # inversion as well).

    # tau = Vrr + Vii = Trace. Guaranteed >= 0 because SPD
    tau = Vrr + Vii
    # delta = (Vrr * Vii) - (Vri ** 2) = Determinant. Guaranteed >= 0 because SPD
    delta = (Vrr * Vii) - (Vri ** 2)

    s = tf.math.sqrt(delta)  # Determinant of square root matrix
    t = tf.math.sqrt(tau + 2 * s)

    # The square root matrix could now be explicitly formed as
    #       [ Vrr+s Vri   ]
    # (1/t) [ Vir   Vii+s ]
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # but we don't need to do this immediately since we can also simultaneously
    # invert. We can do this because we've already computed the determinant of
    # the square root matrix, and can thus invert it using the analytical
    # solution for 2x2 matrices
    #      [ A B ]             [  D  -B ]
    # inv( [ C D ] ) = (1/det) [ -C   A ]
    # http://mathworld.wolfram.com/MatrixInverse.html
    # Thus giving us
    #           [  Vii+s  -Vri   ]
    # (1/s)(1/t)[ -Vir     Vrr+s ]
    # So we proceed as follows:

    inverse_st = 1.0 / (s * t)
    Wrr = (Vii + s) * inverse_st
    Wii = (Vrr + s) * inverse_st
    Wri = -Vri * inverse_st

    # And we have computed the inverse square root matrix W = sqrt(V)!
    # Normalization. We multiply, x_normalized = W.x.

    # The returned result will be a complex standardized input
    # where the real and imaginary parts are obtained as follows:
    # x_real_normed = Wrr * x_real_centred + Wri * x_imag_centred
    # x_imag_normed = Wri * x_real_centred + Wii * x_imag_centred

    broadcast_Wrr = K.reshape(Wrr, variances_broadcast)
    broadcast_Wri = K.reshape(Wri, variances_broadcast)
    broadcast_Wii = K.reshape(Wii, variances_broadcast)

    cat_W_4_real = K.concatenate([broadcast_Wrr, broadcast_Wii], axis=axis)
    cat_W_4_imag = K.concatenate([broadcast_Wri, broadcast_Wri], axis=axis)

    if (axis == 1 and ndim != 3) or ndim == 2:
        centred_real = input_centred[:, :input_dim]
        centred_imag = input_centred[:, input_dim:]
    elif ndim == 3:
        centred_real = input_centred[:, :, :input_dim]
        centred_imag = input_centred[:, :, input_dim:]
    elif axis == -1 and ndim == 4:
        centred_real = input_centred[:, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, input_dim:]
    elif axis == -1 and ndim == 5:
        centred_real = input_centred[:, :, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, :, input_dim:]
    else:
        raise ValueError(
            'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
            'axis: ' + str(axis) + '; ndim: ' + str(ndim) + '.'
        )
    rolled_input = K.concatenate([centred_imag, centred_real], axis=axis)

    output = cat_W_4_real * input_centred + cat_W_4_imag * rolled_input

    #   Wrr * x_real_centered | Wii * x_imag_centered
    # + Wri * x_imag_centered | Wri * x_real_centered
    # -----------------------------------------------
    # = output

    return output


def ComplexBN(input_centred, Vrr, Vii, Vri, beta,
              gamma_rr, gamma_ri, gamma_ii, scale=True,
              center=True, layernorm=False, axis=-1):
    ndim = K.ndim(input_centred)
    input_dim = K.shape(input_centred)[axis] // 2
    if scale:
        gamma_broadcast_shape = [1] * ndim
        gamma_broadcast_shape[axis] = input_dim
    if center:
        broadcast_beta_shape = [1] * ndim
        broadcast_beta_shape[axis] = input_dim * 2

    if scale:
        standardized_output = complex_standardization(
            input_centred, Vrr, Vii, Vri,
            layernorm,
            axis=axis
        )

        # Now we perform th scaling and Shifting of the normalized x using
        # the scaling parameter
        #           [  gamma_rr gamma_ri  ]
        #   Gamma = [  gamma_ri gamma_ii  ]
        # and the shifting parameter
        #    Beta = [beta_real beta_imag].T
        # where:
        # x_real_BN = gamma_rr * x_real_normed + gamma_ri * x_imag_normed + beta_real
        # x_imag_BN = gamma_ri * x_real_normed + gamma_ii * x_imag_normed + beta_imag

        broadcast_gamma_rr = K.reshape(gamma_rr, gamma_broadcast_shape)
        broadcast_gamma_ri = K.reshape(gamma_ri, gamma_broadcast_shape)
        broadcast_gamma_ii = K.reshape(gamma_ii, gamma_broadcast_shape)

        cat_gamma_4_real = K.concatenate(
            [broadcast_gamma_rr, broadcast_gamma_ii], axis=axis)
        cat_gamma_4_imag = K.concatenate(
            [broadcast_gamma_ri, broadcast_gamma_ri], axis=axis)
        if (axis == 1 and ndim != 3) or ndim == 2:
            centred_real = standardized_output[:, :input_dim]
            centred_imag = standardized_output[:, input_dim:]
        elif ndim == 3:
            centred_real = standardized_output[:, :, :input_dim]
            centred_imag = standardized_output[:, :, input_dim:]
        elif axis == -1 and ndim == 4:
            centred_real = standardized_output[:, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, input_dim:]
        elif axis == -1 and ndim == 5:
            centred_real = standardized_output[:, :, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(axis) + '; ndim: ' + str(ndim) + '.'
            )
        rolled_standardized_output = K.concatenate(
            [centred_imag, centred_real], axis=axis)
        if center:
            broadcast_beta = K.reshape(beta, broadcast_beta_shape)
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output + broadcast_beta
        else:
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output
    else:
        if center:
            broadcast_beta = K.reshape(beta, broadcast_beta_shape)
            return input_centred + broadcast_beta
        else:
            return input_centred


class ComplexBatchNormalization(Layer):
    """Complex version of the real domain 
    Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous complex layer at each batch,
    i.e. applies a transformation that maintains the mean of a complex unit
    close to the null vector, the 2 by 2 covariance matrix of a complex unit close to identity
    and the 2 by 2 relation matrix, also called pseudo-covariance, close to the 
    null matrix.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=2` in `ComplexBatchNormalization`.
        momentum: Momentum for the moving statistics related to the real and
            imaginary parts.
        epsilon: Small float added to each of the variances related to the
            real and imaginary parts in order to avoid dividing by zero.
        center: If True, add offset of `beta` to complex normalized tensor.
            If False, `beta` is ignored.
            (beta is formed by real_beta and imag_beta)
        scale: If True, multiply by the `gamma` matrix.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the real_beta and the imag_beta weight.
        gamma_diag_initializer: Initializer for the diagonal elements of the gamma matrix.
            which are the variances of the real part and the imaginary part.
        gamma_off_initializer: Initializer for the off-diagonal elements of the gamma matrix.
        moving_mean_initializer: Initializer for the moving means.
        moving_variance_initializer: Initializer for the moving variances.
        moving_covariance_initializer: Initializer for the moving covariance of
            the real and imaginary parts.
        beta_regularizer: Optional regularizer for the beta weights.
        gamma_regularizer: Optional regularizer for the gamma weights.
        beta_constraint: Optional constraint for the beta weights.
        gamma_constraint: Optional constraint for the gamma weights.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.9,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_diag_initializer='sqrt_init',
                 gamma_off_initializer='zeros',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='sqrt_init',
                 moving_covariance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 beta_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = sanitizedInitGet(beta_initializer)
        self.gamma_diag_initializer = sanitizedInitGet(gamma_diag_initializer)
        self.gamma_off_initializer = sanitizedInitGet(gamma_off_initializer)
        self.moving_mean_initializer = sanitizedInitGet(
            moving_mean_initializer)
        self.moving_variance_initializer = sanitizedInitGet(
            moving_variance_initializer)
        self.moving_covariance_initializer = sanitizedInitGet(
            moving_covariance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        self.beta_constraint = constraints .get(beta_constraint)
        self.gamma_diag_constraint = constraints .get(gamma_diag_constraint)
        self.gamma_off_constraint = constraints .get(gamma_off_constraint)

    def build(self, input_shape):

        ndim = len(input_shape)
        input_shape_real = (input_shape[:self.axis] + input_shape[self.axis]*2)
        dim = input_shape_real[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape_real) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape), shape=input_shape)
        # self.input_spec = InputSpec(ndim=len(input_shape),
        #                             axes={self.axis: dim})

        # param_shape = (input_shape_real[self.axis] ,)
        param_shape = (input_shape_real[self.axis] // 2,)

        if self.scale:
            self.gamma_rr = self.add_weight(shape=param_shape,
                                            name='gamma_rr',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ii = self.add_weight(shape=param_shape,
                                            name='gamma_ii',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ri = self.add_weight(shape=param_shape,
                                            name='gamma_ri',
                                            initializer=self.gamma_off_initializer,
                                            regularizer=self.gamma_off_regularizer,
                                            constraint=self.gamma_off_constraint)
            self.moving_Vrr = self.add_weight(shape=param_shape,
                                              initializer=self.moving_variance_initializer,
                                              name='moving_Vrr',
                                              trainable=False)
            self.moving_Vii = self.add_weight(shape=param_shape,
                                              initializer=self.moving_variance_initializer,
                                              name='moving_Vii',
                                              trainable=False)
            self.moving_Vri = self.add_weight(shape=param_shape,
                                              initializer=self.moving_covariance_initializer,
                                              name='moving_Vri',
                                              trainable=False)
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None
            self.moving_Vrr = None
            self.moving_Vii = None
            self.moving_Vri = None

        if self.center:
            self.beta = self.add_weight(shape=(input_shape_real[self.axis],),
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
            self.moving_mean = self.add_weight(shape=(input_shape_real[self.axis],),
                                               initializer=self.moving_mean_initializer,
                                               name='moving_mean',
                                               trainable=False)
        else:
            self.beta = None
            self.moving_mean = None

        self.built = True

    def call(self, inputs, training=None):
        inputs_real = complex2real(inputs)
        input_shape = K.int_shape(inputs_real)
        ndim = len(input_shape)
        reduction_axes = list(range(ndim))
        del reduction_axes[self.axis]
        input_dim = input_shape[self.axis] // 2
        mu = K.mean(inputs_real, axis=reduction_axes)
        broadcast_mu_shape = [1] * len(input_shape)
        broadcast_mu_shape[self.axis] = input_shape[self.axis]
        broadcast_mu = K.reshape(mu, broadcast_mu_shape)
        if self.center:
            input_centred = inputs_real - broadcast_mu
        else:
            input_centred = inputs_real
        centred_squared = input_centred ** 2
        if (self.axis == 1 and ndim != 3) or ndim == 2:
            centred_squared_real = centred_squared[:, :input_dim]
            centred_squared_imag = centred_squared[:, input_dim:]
            centred_real = input_centred[:, :input_dim]
            centred_imag = input_centred[:, input_dim:]
        elif ndim == 3:
            centred_squared_real = centred_squared[:, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, input_dim:]
            centred_real = input_centred[:, :, :input_dim]
            centred_imag = input_centred[:, :, input_dim:]
        elif self.axis == -1 and ndim == 4:
            centred_squared_real = centred_squared[:, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, input_dim:]
        elif self.axis == -1 and ndim == 5:
            centred_squared_real = centred_squared[:, :, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        if self.scale:
            Vrr = K.mean(
                centred_squared_real,
                axis=reduction_axes
            ) + self.epsilon
            Vii = K.mean(
                centred_squared_imag,
                axis=reduction_axes
            ) + self.epsilon
            # Vri contains the real and imaginary covariance for each feature map.
            Vri = K.mean(
                centred_real * centred_imag,
                axis=reduction_axes,
            )
        elif self.center:
            Vrr = None
            Vii = None
            Vri = None
        else:
            raise ValueError(
                'Error. Both scale and center in batchnorm are set to False.')

        input_bn = ComplexBN(
            input_centred, Vrr, Vii, Vri,
            self.beta, self.gamma_rr, self.gamma_ri,
            self.gamma_ii, self.scale, self.center,
            axis=self.axis
        )
        input_bn = real2complex(input_bn)
        if training in {0, False}:
            return input_bn
        else:
            update_list = []
            if self.center:
                update_list.append(K.moving_average_update(
                    self.moving_mean, mu, self.momentum))
            if self.scale:
                update_list.append(K.moving_average_update(
                    self.moving_Vrr, Vrr, self.momentum))
                update_list.append(K.moving_average_update(
                    self.moving_Vii, Vii, self.momentum))
                update_list.append(K.moving_average_update(
                    self.moving_Vri, Vri, self.momentum))
            self.add_update(update_list, inputs_real)

            def normalize_inference():
                if self.center:
                    inference_centred = inputs_real - \
                        K.reshape(self.moving_mean, broadcast_mu_shape)
                else:
                    inference_centred = inputs_real
                inference_bn = ComplexBN(
                    inference_centred, self.moving_Vrr, self.moving_Vii,
                    self.moving_Vri, self.beta, self.gamma_rr, self.gamma_ri,
                    self.gamma_ii, self.scale, self.center, axis=self.axis
                )
                return real2complex(inference_bn)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(input_bn,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer':              sanitizedInitSer(self.beta_initializer),
            'gamma_diag_initializer':        sanitizedInitSer(self.gamma_diag_initializer),
            'gamma_off_initializer':         sanitizedInitSer(self.gamma_off_initializer),
            'moving_mean_initializer':       sanitizedInitSer(self.moving_mean_initializer),
            'moving_variance_initializer':   sanitizedInitSer(self.moving_variance_initializer),
            'moving_covariance_initializer': sanitizedInitSer(self.moving_covariance_initializer),
            'beta_regularizer':              regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer':        regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer':         regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint':               constraints .serialize(self.beta_constraint),
            'gamma_diag_constraint':         constraints .serialize(self.gamma_diag_constraint),
            'gamma_off_constraint':          constraints .serialize(self.gamma_off_constraint),
        }
        base_config = super(ComplexBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
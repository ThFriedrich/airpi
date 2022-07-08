import ap_training.losses as ls
from copy import copy
import tensorflow as tf
from ap_architectures.utils import tf_pi, tf_sin_cos2rad
from ap_utils.util_fcns import PRM
from tensorflow.signal import ifft2d, fftshift, ifftshift
import tensorflow.keras.layers as kl 
from tensorflow.keras import Model
from tf_complex.convolutions import ComplexConv2D
from tensorflow import keras as tfk
kl = tfk.layers
from ap_architectures.bn import ComplexBatchNormalization
class CNET():
    # Derived from https://arxiv.org/abs/2004.01738
    def __init__(self, x_shape, prms):

        # Architecture parameters
        self.x_shape = copy(x_shape)
        if 'deploy' in prms and prms['deploy']:
            self.deploy = True
        else:
            self.deploy = False

        self.type = prms['type']
        self.kernel = prms['kernel']
        self.filters = prms['filters']
        self.depth = prms['depth']
        self.normalization = int(prms['normalization'])
        self.activation = 'cardioid'
        self.stack_n = prms['stack_n']
        # Regularization parameters

    def flatten(self, x: tf.Tensor) -> tf.Tensor:
        return x**0.1

    def standardize(self, x: tf.Tensor) -> tf.Tensor:
        x_mean, x_var = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_std = tf.maximum(tf.sqrt(x_var), 1e-8)
        x = tf.divide(tf.subtract(x, x_mean), x_std)

        return x, x_mean, x_std


    def x_normalise(self, x: tf.Tensor) -> tf.Tensor:
        # xf = self.flatten(xf)
        xf, x_mean, x_std = self.standardize(self.flatten(x[..., 0:9]))
        if x.shape[-1] > 9:
            msk_a = self.standardize(tf.expand_dims(x[..., 9], -1))
            xf = tf.concat((xf, msk_a[0]), -1)
        if x.shape[-1] > 10:
            msk_p = self.standardize(
                tf.stack((tf.sin(x[..., 10]), tf.cos(x[..., 10])), -1))
            xf = tf.concat((xf, msk_p[0]), -1)
        return x_mean, x_std, xf


    # Input Layer
    def input_block(self):
        x0 = kl.Input(
            shape=(self.x_shape[0], self.x_shape[1], self.x_shape[2]), name='cbeds')
        x = x0
        # if self.deploy:
        # x = tf.image.resize(x, [64, 64])
        x_mean, x_std, x = self.x_normalise(x)
        x_mean2 = tf.math.reduce_mean(x0)
        return x_mean, x_std, x, x0, x_mean2

    def output_amplitude(self, x):
        x = ComplexConv2D(2, self.kernel, padding='same', activation='linear')(x)
        x = tf.math.abs(x) * 1e-3
        # x = tf.math.abs(x)
        # x = tfk.activations.sigmoid(x)
        return x

    def output_phase(self, x):
        x = ComplexConv2D(2, self.kernel, padding='same', activation='linear')(x)
        x = tf.math.angle(x)
        return x, tf.sin(x), tf.cos(x)

    def complex2real(self,x):
        x_real = tf.math.real(x)
        x_imag = tf.math.imag(x)
        return tf.concat([x_real,x_imag], axis=-1)

    def real2complex(self, x):
        channel = x.shape[-1] // 2
        if x.shape.ndims == 3:
            return tf.complex(x[:,:,:channel], x[:,:,channel:])
        elif x.shape.ndims == 4:
            return tf.complex(x[:,:,:,:channel], x[:,:,:,channel:])

    def conv_norm_layer(self, x, nf, strides=[1,1]):
        x = ComplexConv2D(nf, self.kernel, strides=strides, padding='same', activation=self.activation)(x)
        if self.normalization == 1:
            x = ComplexBatchNormalization()(x)
        return x

    def conv_stack(self, x):
        nf = x.shape[-1]*2
        for _ in range(self.stack_n):
            x = self.conv_norm_layer(x, nf)
        return x

    def down_block(self,x):
        nf = x.shape[-1]
        sk = self.conv_stack(x)
        x = self.conv_norm_layer(sk, nf*4, strides=[2,2])
        # x = ComplexConv2D(nf*fct*2, self.kernel, strides=[2,2], padding='same', activation=self.activation)(sk)

        return x, sk

    def up_block(self, x, x_skip):
        x = tf.nn.depth_to_space(x, 2)
        x = kl.concatenate([x, x_skip])
        x = self.conv_stack(x)
        return x

    def oversample(self, array):
        return tf.pad(array, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])

    def ifft2d(self, array):
        return fftshift(ifft2d(ifftshift(array[..., 0])))

    def deploy_out(self, x_i, x_p):
        x_i = tf.cast(x_i**5, tf.complex64)
        x_p = tf.cast(x_p, tf.complex64)
        x_o = (x_i * tf.exp(1j*x_p))
        return x_o

    def build(self):

        nf = self.depth -1

        x_mean, x_std, x, x0, x_mean2 = self.input_block()
        skips = []

        x = ComplexConv2D(self.filters*2, self.kernel, padding='same', activation=self.activation)(x)
        for _ in range(nf):
            x, sk = self.down_block(x)
            skips.append(sk)
        x = self.conv_stack(x)
        for ix in range(nf):
            x = self.up_block(x, skips[-(ix+1)])

        x_i = self.output_amplitude(x)
        x_p, x_s, x_c = self.output_phase(x)

        if self.deploy:
            x_o = self.deploy_out(x_i, x_p)
            pos = kl.Input(shape=(2), dtype=tf.int32, name='pos')
            model = Model(inputs=[x0, pos], outputs=[x_o, pos])
        else:
            x_o = kl.Concatenate(dtype=tf.float32)((x_i, x_s, x_c))
            model = Model(inputs=x0, outputs=x_o)
            model.summary()

        return model

   
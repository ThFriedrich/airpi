import airpi.ap_training.losses as ls
from copy import copy
import tensorflow as tf
from airpi.ap_architectures.utils import tf_pi_32, fcn_rad_CS, PRM, floatx, tf_butterworth_filter2D
from tensorflow.signal import ifft2d, fftshift, ifftshift
import tensorflow.keras.layers as kl 
from tensorflow.keras import Model
from tf_complex.convolutions import ComplexConv2D
from tensorflow import keras as tfk
kl = tfk.layers

class CNET():
    # Derived from https://arxiv.org/abs/2004.01738
    def __init__(self, x_shape, prms, scaled_out_size=None):

        # Architecture parameters
        self.x_shape = copy(x_shape)
        if 'predict' in prms and prms['predict']:
            # self.x_shape = self.x_shape.as_list()
            self.deploy = True
            if scaled_out_size is None: scaled_out_size = x_shape[0]
            self.pad = int(scaled_out_size - x_shape[0]) // 2
            self.btw_filter = tf.expand_dims(tf.cast(tf_butterworth_filter2D(
                [scaled_out_size, scaled_out_size], 0.025, order=12, shape='ci'), tf.complex64), -1)
        else:
            self.deploy = False
            if prms['bin_mask']:
                self.x_shape[2] += PRM.mask_num

        self.type = prms['type']
        self.kernel = prms['kernel']
        self.filters = prms['filters']
        self.depth = prms['depth']
        self.normalization = int(prms['normalization'])
        self.activation = prms['activation']
        self.stack_n = prms['conv_stack_n']
        # Regularization parameters

    def flatten(self, x: tf.Tensor) -> tf.Tensor:
        return x**0.1

    def standardize(self, x: tf.Tensor) -> tf.Tensor:
        x_mean, x_var = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_std = tf.maximum(tf.sqrt(x_var), 1e-8)
        x = tf.divide(tf.subtract(x, x_mean), x_std)
        return x_mean, x_std, x

    def x_normalise(self, x: tf.Tensor) -> tf.Tensor:
        # xf = self.flatten(xf)
        x_mean, x_std, xf = self.standardize(self.flatten(x[..., 0:9]))
        xf = tf.cast(xf, tf.complex64)
        if x.shape[-1] > 9:
            xp = tf.cast(x[..., 9:], tf.complex64)
            self.probe = xp[..., 0]*tf.exp(1j*xp[..., 1])
            xf = tf.concat((xf, self.probe[...,tf.newaxis]), -1)
        return x_mean, x_std, xf


    # Input Layer
    def input_block(self):
        x0 = kl.Input(
            shape=(self.x_shape[0], self.x_shape[1], self.x_shape[2]), name='cbeds', dtype=floatx)
        x = x0
        # if self.deploy:
        # x = tf.image.resize(x, [64, 64])
        x_mean, x_std, x = self.x_normalise(x)

        return x_mean, x_std, x, x0

    def output_amplitude(self, x):
        x = ComplexConv2D(2, self.kernel, padding='same', activation='linear')(x)
        return tf.math.abs(x)

    def output_phase(self, x):
        x = ComplexConv2D(2, self.kernel, padding='same', activation='linear')(x)
        x = tf.math.angle(x)
        return x, tf.sin(x), tf.cos(x)

    def conv_stack(self, x):
        nf = x.shape[-1]*2
        for _ in range(self.stack_n):
            x = ComplexConv2D(nf, self.kernel, padding='same', activation=self.activation)(x)
        return x

    def down_block(self,x):
        nf = x.shape[-1]
        fct = 2
        sk = self.conv_stack(x)
        x = ComplexConv2D(nf*fct*2, self.kernel, strides=[fct,fct], padding='same', activation=self.activation)(sk)
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

    def build(self):

        nf = self.depth -1

        x_mean, x_std, x, x0 = self.input_block()
        skips = []

        x = ComplexConv2D(self.filters*2, self.kernel, padding='same', activation='crelu')(x)
        for _ in range(nf):
            x, sk = self.down_block(x)
            skips.append(sk)
        x = self.conv_stack(x)
        for ix in range(nf):
            x = self.up_block(x, skips[-(ix+1)])

        # x_i = self.output_amplitude(x)
        # x_p, x_s, x_c = self.output_phase(x)
        x_o = ComplexConv2D(2, self.kernel, padding='same', activation='linear')(x)
        # x_o = kl.layers.Concatenate([tf.math.real(x_o), tf.math.imag(x_o)])
        x_o = tf.concat([tf.math.real(x_o), tf.math.imag(x_o)], -1)
        if self.deploy:
            x_o = tf.complex(x_o[..., 0], x_o[..., 1])
            # x_i = tf.math.add(x_mean,x_i)**5
            # x_i = (x_i/x_mean)**5
            # x_i = (tf.maximum(0.00, tf.math.add(x_mean,x_i)))**5
            # x_i = tf.cast(x_i, tf.complex64)
            # x_p = tf.cast(x_p, tf.complex64)
            # x_o = x_i * tf.exp(1j*x_p)
            # x_o = tf.where(tf.abs(self.probe[...,tf.newaxis])>0,x_o,0)
            # x_o = self.oversample(x_o)
            # x_o *=  self.btw_filter
            # x_o = self.ifft2d(x_o)
            # x_o = tfk.layers.Concatenate(dtype=tf.float32)((x_i, x_p))
            pos = kl.Input(shape=(2), dtype=tf.int32, name='pos')
            model = Model(inputs=[x0, pos], outputs=[x_o, pos])
        else:
            # x_o = kl.Concatenate(dtype=tf.float32)((x_i, x_s, x_c))
            model = Model(inputs=x0, outputs=x_o)

        model.summary()

        return model

   
import ap_training.losses as ls
from copy import copy
import tensorflow as tf
from ap_architectures.utils import tf_binary_mask, tf_pi, tf_sin_cos2rad, floatx
from ap_utils.util_fcns import PRM
from tensorflow import keras as tfk
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization
from tensorflow.signal import ifft2d, fftshift, ifftshift
kl = tfk.layers

class amplitude_output(tfk.layers.Layer):
  """Layer that creates an activity sparsity regularization loss."""

  def __init__(self,b_skip=True):
    super(amplitude_output, self).__init__()
    self.scale = None
    self.b_skip = b_skip
    self.conv = kl.Conv2D(1, kernel_size=[3, 3], padding='same',
                      strides=[1, 1], kernel_regularizer=None, activity_regularizer=None,
                       activation='linear', dtype=tf.float32)

  def df_bf_ratio_loss(self, x0, x_i):
    b_msk = tf.cast(x0[...,9],tf.bool)
    bf_t = tf.math.reduce_mean(tf.where(b_msk, x0[...,4], 0), axis=[1, 2])
    df_t = tf.math.reduce_mean(tf.where(~b_msk, x0[...,4], 0),axis=[1,2])
    ratio_t = tf.math.maximum(tf.math.divide_no_nan(df_t,bf_t), 1e-7)
    bf_p = tf.math.reduce_mean(tf.where(b_msk,tf.squeeze(x_i), 0), axis=[1, 2])
    df_p = tf.math.reduce_mean(tf.where(~b_msk,tf.squeeze(x_i), 0),axis=[1,2])
    ratio_p =  tf.math.maximum(tf.math.divide_no_nan(df_p,bf_p), 1e-7)
    ls =  tf.math.reduce_mean(tf.math.abs(ratio_p - ratio_t) / ratio_t) * 1e-5
    
    self.add_loss(ls)
    self.add_metric(ls, name='df_ratio_k2')

  def call(self, x0, x):
    if self.scale is None:
      self.scale = tf.Variable(initial_value=0.1, dtype=tf.float32, trainable=True)

    x = self.conv(x)
    if self.b_skip:
        dose = tf.reduce_sum(x0[...,4]) 
        x_add = (x0[..., 4, tf.newaxis]/dose)**0.05
        x = kl.add([x, x_add])

    self.add_metric(self.scale, name='amp_scale')
    x = tf.maximum(0.0,x) * self.scale
    # x = tf.math.sigmoid(x)
    self.df_bf_ratio_loss(x0, x)

    return x
class phase_output(tfk.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, n=1):
        super(phase_output, self).__init__()
        self.n = n
        self.conv = kl.Conv2D(self.n, kernel_size=[3, 3], padding='same',
                            strides=[1, 1], kernel_regularizer=None, activity_regularizer=None,
                            activation='linear', dtype=tf.float32)

    def decompose(self, x):
        if self.n > 1:
            x_sin, x_cos = tf.split(x)
            x_sin = tf.math.sin(x_sin)
            x_cos = tf.math.cos(x_cos)
            ls = tf.norm(tf.stack([x_sin, x_cos]), ord="euclidean", axis=0) - 1.0
            self.add_loss(ls)
            self.add_metric(ls, name='sin_cos_ecn')
        else:
            x_sin = tf.sin(x)
            x_cos = tf.cos(x)
        return x_sin, x_cos

    def call(self, x):
        x = self.conv(x)
        x = tf.math.tanh(x)*tf_pi
        x_sin, x_cos = self.decompose(x)

        return x_sin, x_cos


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
            msk_a = self.standardize(x[..., 9,tf.newaxis])
            xf = tf.concat((xf, msk_a), -1)
        if x.shape[-1] > 10:
            msk_p = self.standardize(
                tf.stack((tf.sin(x[..., 10]), tf.cos(x[..., 10])), -1))
            xf = tf.concat((xf, msk_p[0]), -1)
        return xf

    def call(self, x):
        return self.x_normalise(x)

class Grouped_Batch_Normalization(tfk.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, groups=2):
        super(Grouped_Batch_Normalization, self).__init__()
        self.groups = groups
        self.batch_norm = []
        for _ in range(groups):
            self.batch_norm.append(kl.BatchNormalization())
   

    def normalise(self, x: tf.Tensor, training:bool) -> tf.Tensor:
        grps = tf.split(x,self.groups,-1)
        for ig, g in enumerate(grps):
            grps[ig] = self.batch_norm[ig](g,training)
        x = tf.concat(grps,-1)
        return x

    def call(self, x, training=None):
        return self.normalise(x,training)


class UNET():
    # Derived from https://arxiv.org/pdf/1505.04597.pdf
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
        self.activation = int(prms['activation'])
        self.stack_n = prms['stack_n']
        
        # Regularization parameters
        if prms['w_regul'] != None:
            self.w_regul = tfk.regularizers.l2(float(prms['w_regul']))
        else:
            self.w_regul = None
        if prms['a_regul'] != None:
            self.a_regul = tfk.regularizers.l1(float(prms['a_regul']))
        else:
            self.a_regul = None
        self.dropout = prms['dropout']

        # Initialization
        self.kernel_initializer = tfk.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal', seed=None)

        self.vb = prms['branching']

        if 'hp_search' in prms:
            self.loss_fcn = ls.loss(prms)
            self.metric_fcn = ls.metric_fcns(prms)


  
    def safe_softplus(self, x, limit=30):
        return tf.where(x > limit, x, tf.math.log1p(tf.exp(x)))

    def safe_flat_softplus(self, x, nu=0, limit=30):
        return tf.where(x > limit, x, tf.math.log1p(tf.exp(x*(1-nu))))

    def normalization_layer(self, x, nor=-1):
        if nor < 0:
            nor = self.normalization
        if nor == 0:
            pass
        elif nor == 1:
            x = InstanceNormalization()(x)
        elif nor == 2:
            x = kl.LayerNormalization()(x)
        elif nor == 3:
            x = Grouped_Batch_Normalization()(x)
        elif nor == 7:
            x = GroupNormalization(groups=2)(x)
        else:
            x = kl.BatchNormalization()(x)
        return x

    def activation_layer(self, x, act=-1):
        if act < 0:
            act = self.activation
        if act == 0:
            pass
        elif act == 1:
            x = kl.LeakyReLU()(x)
        elif act == 2:
            x = kl.ELU()(x)
        elif act == 3:
            x = tfk.activations.swish(x)
        elif act == 6:
            x = self.safe_softplus(x)
        else:
            x = kl.ReLU()(x)
        return x

    def convolution_layer(self, x, nf, strides=[1, 1]):
        x = kl.Conv2D(nf, kernel_size=self.kernel, padding='same',
                      strides=strides, kernel_regularizer=self.w_regul, activity_regularizer=self.a_regul,
                      activation=None)(x)
        return x

    def conv_normalization_layer(self, x, nf, act=-1, nor=-1, strides=[1, 1]):
        x = self. convolution_layer(x, nf, strides)
        x = self.normalization_layer(x, nor)
        x = self.activation_layer(x, act)
        if self.dropout is not None:
            x = kl.Dropout(self.dropout)(x)

        return x

    # Input Layer
    def input_block(self):
        x0 = kl.Input(
            shape=(self.x_shape[0], self.x_shape[1], self.x_shape[2]), name='cbeds', dtype=floatx)
        x = x0
        # if self.deploy:
        #     x = tf.image.resize(x, [64, 64])
        x_mean, x_std, x = self.x_normalise(x)
        dose = tf.reduce_sum(x0[...,4])
        return x_mean, x_std, x, x0, dose

    def dcr_block(self, x, act=-1, nor=-1):
        nf = x.shape[-1]
        x1 = self.conv_normalization_layer(x, (nf*1.5), act, nor)
        x1 = kl.concatenate([x, x1])

        x2 = self.conv_normalization_layer(x1, (nf*2), act, nor)
        x2 = kl.concatenate([x, x2])

        x3 = self.conv_normalization_layer(x2, nf, act, nor)

        x3 = kl.add((x, x3))

        return x3

    def dcr_stack(self, x, act=-1, nor=-1):
        c1 = self.dcr_block(x, act, nor)
        c = kl.add((x, c1))
        c2 = self.dcr_block(c1, act, nor)
        c = kl.add((c1, c2))
        return c

    def res_stack(self, x, act=-1, nor=-1):
        nf = x.shape[-1]
        x2 = self.conv_normalization_layer(x, nf, act, nor)
        for _ in range(self.stack_n-2):
            x2 = self.conv_normalization_layer(x2, nf, act, nor)
        x = self.convolution_layer(x, nf)
        x = kl.add((x, x2))
        x = self.normalization_layer(x)
        x = self.activation_layer(x)

        return x

    def conv_stack(self, x, act=-1, nor=-1):
        nf = x.shape[-1]
        for _ in range(self.stack_n):
            x = self.conv_normalization_layer(x, nf, act, nor)

        return x

    def down_block(self, x, act=-1, nor=-1, t='V'):
        nf = x.shape[-1]
        fct = 2
        if t == 'DCR':
            c = self.dcr_stack(x)
            fct = 2
        elif t == 'RES':
            c = self.res_stack(x)
        else:
            c = self.conv_stack(x, act, nor)
        # x = self.conv_normalization_layer(
            # c, (x.shape[-1]*4), act, nor, strides=[4, 4])
        x = self.conv_normalization_layer(
            c, (int(fct*x.shape[-1])), act, nor, strides=[2, 2])
        return x, c

    def up_block(self, x, x_skip, act=-1, nor=-1, t='V'):
        nf = x_skip.shape[-1]
        x = kl.Conv2DTranspose(
            # nf, self.kernel, strides=(4, 4), padding='same')(x)
            nf, self.kernel, strides=(2, 2), padding='same')(x)
        x = kl.concatenate([x, x_skip])
        if t == 'DCR':
            x = self.dcr_stack(x)
        elif t == 'RES':
            x = self.res_stack(x)
        else:
            x = self.conv_stack(x, act, nor)

        return x

    def output_amplitude(self, x, n_out, act='linear'):
        x = kl.Conv2D(n_out, kernel_size=self.kernel, padding='same',
                      strides=[1, 1], kernel_regularizer=self.w_regul, activity_regularizer=self.a_regul,
                       activation='linear', dtype=tf.float32)(x)
        a_fct = tf.Variable(1e-3, dtype=tf.float32, trainable=True)
        # x = kl.LeakyReLU()(x)
        # x = kl.ReLU()(x)
        # x = self.safe_flat_softplus(x, nu=0.5, limit=30)
        x = tf.maximum(0.0,x) * a_fct
        # x = self.safe_softplus(x) 
        # x = tfk.activations.sigmoid(x)
        # x = tfk.activations.relu(x) * 1e-3
        return x

    def output_phase_decomposition(self, x, act='linear'):
        x = kl.Conv2D(1, kernel_size=self.kernel, padding='same',
                      strides=[1, 1], kernel_regularizer=self.w_regul, activity_regularizer=self.a_regul,
                       activation=act, dtype=tf.float32)(x)
        x_sin = tf.sin(x)
        x_cos = tf.cos(x)
        return x_sin, x_cos

    # def output_phase_decomposition(self, x, act='linear'):
    #     x = kl.Conv2D(2, kernel_size=self.kernel, padding='same',
    #                   strides=[1, 1], kernel_regularizer=self.w_regul, activity_regularizer=self.a_regul,
    #                    activation=act, dtype=tf.float32)(x)
    #     x_sin, x_cos = self.phase_decomposition(x)

    #     return x_sin, x_cos

    def phase_decomposition(self, x):
        x = tf.math.tanh(x) * tf_pi
        x_sin, x_cos = tf.split(x, 2, axis=-1)
        x_sin = tf.sin(x_sin)
        x_cos = tf.cos(x_cos)
        return x_sin, x_cos

    def oversample(self, array):
        return tf.pad(array, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])

    def ifft2d(self, array):
        return fftshift(ifft2d(ifftshift(array[..., 0])))

    def deploy_out(self, x_i, x_s, x_c):
        x_i = tf.cast(x_i**5, tf.complex64)
        x_p = tf.cast(tf_sin_cos2rad(x_s, x_c), tf.complex64)
        x_o = (x_i * tf.exp(1j*x_p))
        return x_o

    def build_fix(self):

        nf = self.depth-1
        c = list()
        x0 = kl.Input(shape=(self.x_shape[0], self.x_shape[1], self.x_shape[2]), name='cbeds')
        x = Standardization_Layer()(x0)
        # x_mean, x_std, x, x0, dose = self.input_block()
        # x_add = (x0[..., 4, tf.newaxis])
        # x_add = (x0[..., 4, tf.newaxis]/dose)**0.05

        x = self.conv_normalization_layer(x, self.filters)

        for ix in range(0, nf):
            x, ct = self.down_block(x,t=self.type)
            c.append(ct)

        if self.type == 'DCR':
            x = self.dcr_stack(x)
        elif self.type == 'RES':
            x = self.res_stack(x)
        else:
            x = self.conv_stack(x)

        for ix in range(0, nf):
            x = self.up_block(x, c[-(ix+1)],t=self.type)

        # x_i = (x_i + tf.math.sqrt(x_mean2))
        # x_i = self.output_amplitude(x, 1)
        # x_i = kl.add([x_i, x_add])
        # x_i = x_add
        # x_i = DF_BF_Ratio()(x0, x_i)

        # x_s, x_c = self.output_phase_decomposition(x)
        x_i = amplitude_output(b_skip=True)(x0,x)
        x_s, x_c = phase_output()(x)

        if self.deploy:
            x_o = self.deploy_out(x_i, x_s, x_c)
            pos = kl.Input(shape=(2), dtype=tf.int32, name='pos')
            model = tfk.Model(inputs=[x0, pos], outputs=[x_o, pos])
        else:
            x_o = tfk.layers.Concatenate(dtype=tf.float32)((x_i, x_s, x_c))
            model = tfk.Model(inputs=x0, outputs=x_o)
            model.summary()

        return model

    def build_fix_branched_vb1(self):

        nf = self.depth-1
        c1 = list()
        c2 = list()

        x_mean, x_std, x, x0, x_mean2 = self.input_block()
        x1 = x[..., 0:10]
        # x_add = (x[..., 4, tf.newaxis])
        x_add = (x0[..., 4, tf.newaxis]/x_mean2)**0.1

        x2 = tf.concat([x[..., 0:9], x[..., 10:]], axis=-1)

        x1 = self.conv_normalization_layer(x1, self.filters)
        x2 = self.conv_normalization_layer(x2, self.filters)

        for ix in range(0, nf):
            x1, ct1 = self.down_block(x1, t='V')
            c1.append(ct1)
            x2, ct2 = self.down_block(x2, t='V')
            c2.append(ct2)

        x1 = self.conv_stack(x1)
        x2 = self.conv_stack(x2)

        for ix in range(0, nf):
            x1 = self.up_block(x1, c1[-(ix+1)], t='V')
            x2 = self.up_block(x2, c2[-(ix+1)], t='V')

        x1 = kl.add([x1, x_add])
        x1 = self.conv_stack(x1)

        x2 = tf.concat([x2, x1], axis=-1)
        x2 = self.conv_stack(x2)

        x_i = self.output_amplitude(x1, 1)
        x_s, x_c = self.output_phase_decomposition(x2)

        if self.deploy:
            x_o = self.deploy_out(x_i, x_s, x_c)
            pos = kl.Input(shape=(2), dtype=tf.int32, name='pos')
            model = tfk.Model(inputs=[x0, pos], outputs=[x_o, pos])
        else:
            x_o = tfk.layers.Concatenate(dtype=tf.float32)((x_i, x_s, x_c))
            model = tfk.Model(inputs=x0, outputs=x_o)
            model.summary()


        return model

    def build_fix_branched_vb2(self):

        nf = self.depth-1
        c1 = list()
        c2 = list()

        x_mean, x_std, x, x0, x_mean2 = self.input_block()
        # x_add = x[..., 4, tf.newaxis]
        x_add = (x0[..., 4, tf.newaxis]/x_mean2)**0.1

        x1 = x[..., 0:10]
        x2 = tf.concat([x[..., 0:9], x[..., 10:]], axis=-1)

        x1 = self.conv_normalization_layer(x1, self.filters)
        x2 = self.conv_normalization_layer(x2, self.filters)

        for ix in range(0, nf):
            x1, ct1 = self.down_block(x1, t='RES')
            c1.append(ct1)
            x2, ct2 = self.down_block(x2, t='V')
            c2.append(ct2)

        x1 = self.res_stack(x1)
        x2 = self.conv_stack(x2)

        for ix in range(0, nf):
            x1 = self.up_block(x1, c1[-(ix+1)], t='RES')
            x2 = self.up_block(x2, c2[-(ix+1)], t='V')

        x1 = kl.add([x1, x_add])
        x1 = self.res_stack(x1)

        x2 = tf.concat([x2, x1], axis=-1)
        x2 = self.conv_stack(x2)

        x_i = self.output_amplitude(x1, 1)
        x_s, x_c = self.output_phase_decomposition(x2)

        if self.deploy:
            x_o = self.deploy_out(x_i, x_s, x_c)
            pos = kl.Input(shape=(2), dtype=tf.int32, name='pos')
            model = tfk.Model(inputs=[x0, pos], outputs=[x_o, pos])
        else:
            x_o = tfk.layers.Concatenate(dtype=tf.float32)((x_i, x_s, x_c))
            model = tfk.Model(inputs=x0, outputs=x_o)
            model.summary()

        return model

    def build_optim(self, hp):

        act = 1  # hp.Choice('act',[0, 1, 2, 3, 4])
        self.type = 'V'  # hp.Choice('type',['RES','V'])
        depth = 3  # hp.Int('depth', 2, 5, step=1)
        filters = 32  # hp.Int('filters', 16, 48, step=8)
        prms = dict()

        lr = hp.Float('learning_rate', 1e-6, 1e-4, sampling='log')

        c = list()

        x, x0 = self.input_block()
        x = self.conv_normalization_layer(x, filters)

        for ix in range(0, depth):
            x, ct = self.down_block(x, act=act)
            c.append(ct)

        if self.type == 'DCR':
            x = self.dcr_stack(x, act=act)
        elif self.type == 'RES':
            x = self.res_stack(x, act=act)
        else:
            x = self.conv_stack(x, act=act)

        for ix in range(0, depth):
            x = self.up_block(x, c[-(ix+1)], act=act)

        x_i = self.output_amplitude(x, 1)
        x_s, x_c = self.output_phase_decomposition(x)

        x_o = tfk.layers.Concatenate(dtype=tf.float32)((x_i, x_s, x_c))
        model = tfk.Model(inputs=x0, outputs=x_o)

        prms['pred_space'] = 'k'
        prms['loss_prms_r'] = [1.0]
        prms['loss_prms_k'] = [1.0]
        for ix in range(1, 5):
            prms['loss_prms_r'].append(
                hp.Float('loss_r_'+str(ix), min_value=0.0, max_value=5.0))
            prms['loss_prms_k'].append(
                hp.Float('loss_k_'+str(ix), min_value=0.0, max_value=5.0))
        prms['loss_prms_r'].append(0.0)
        prms['loss_prms_k'].append(0.0)
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(lr),
            loss=ls.loss(prms),
            metrics=[self.metric_fcn])

        return model

    def build(self):
        if self.vb == 0:
            return self.build_fix()
        elif self.vb == 1:
            return self.build_fix_branched_vb1()
        elif self.vb == 2:
            return self.build_fix_branched_vb2()

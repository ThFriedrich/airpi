from tensorflow import keras as tfk
from ap_architectures.layers import *
import tensorflow as tf
from ap_architectures.layers_complex import *


class UNET(tfk.Model):
    def __init__(self, prms, input_shape, deploy=False, **kwargs):
        super(UNET, self).__init__(**kwargs)
        self.x_shape = input_shape
        self.type = prms['type']
        self.kernel = prms['kernel']
        self.filters = prms['filters']
        self.depth = prms['depth']
        self.normalization = int(prms['normalization'])
        self.activation = int(prms['activation'])
        self.stack_n = prms['stack_n']
        self.global_skip = bool(prms['global_skip'])
        self._name = "UNET_" + str(self.filters) + "_D" + str(self.depth) + \
            "_N" + str(self.normalization) + "_A" + str(self.activation)

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
        self.deploy = deploy
        # Initialization
        self.kernel_initializer = tfk.initializers.HeNormal(seed=1310)
        self.Input = kl.Input(
            shape=(self.x_shape[0], self.x_shape[1], self.x_shape[2]), name='cbeds')
        self.Standardization_Layer = Standardization_Layer()
        self.Convolution_Block = Convolution_Block(filters=self.filters, kernel_size=self.kernel, initializer=self.kernel_initializer, kernel_regularizer=self.w_regul,
                                                   activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation)
        self.Contraction_Block = []
        self.Expansion_Block = []
        self.Amplitude_Output = Amplitude_Output(b_skip=self.global_skip)
        self.Phase_Output = Phase_Output(b_skip=self.global_skip)
        self.Conv_Stack = Conv_Stack(self.stack_n, kernel_size=self.kernel, initializer=self.kernel_initializer, kernel_regularizer=self.w_regul,
                                     activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation)
        for _ in range(self.depth):
            self.Contraction_Block.append(Contraction_Block(initializer=self.kernel_initializer, kernel_regularizer=self.w_regul,
                                          activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation))
            self.Expansion_Block.append(Expansion_Block(initializer=self.kernel_initializer, kernel_regularizer=self.w_regul,
                                        activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation))
        self.Deploy_Output = Deploy_Output()

    def build(self):
        super().build(input_shape=(
            None, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
        return self

    def call(self, inputs, training=False):

        x = self.Standardization_Layer(inputs)
        x = self.Convolution_Block(x)

        skips = list()
        for i in range(self.depth):
            x, ct = self.Contraction_Block[i](x)
            skips.append(ct)

        for i in range(self.depth):
            x = self.Expansion_Block[i](x, skips[-(i+1)])

        x_a = self.Amplitude_Output(inputs, x)
        x_p = self.Phase_Output(inputs, x)
        x_o = tf.concat((x_a, x_p), -1)

        if self.deploy:
            return self.Deploy_Output(x_o)
        else:
            return x_o

    def summary(self):
        tfk.Model(inputs=self.Input, outputs=self.call(self.Input)).summary()


class CNET(tfk.Model):
    def __init__(self, prms, input_shape, deploy=False, **kwargs):
        super(CNET, self).__init__(**kwargs)
        self.x_shape = input_shape
        self.kernel = prms['kernel']
        self.filters = prms['filters']
        self.depth = prms['depth']
        self.normalization = int(prms['normalization'])
        self.activation = int(prms['activation'])
        self.stack_n = prms['stack_n']
        self.global_skip = bool(prms['global_skip'])
        self.dropout = prms['dropout']
        self._name = "CNET_" + str(self.filters) + "_D" + str(self.depth) + \
            "_N" + str(self.normalization) + "_A" + str(self.activation)

        self.deploy = deploy
        # Initialization
        self.Input = kl.Input(
            shape=(self.x_shape[0], self.x_shape[1], self.x_shape[2]), name='cbeds')
        self.Standardization_Layer = Standardization_Layer()
        self.Convolution_Block = Convolution_Block_Complex(
            filters=self.filters, kernel_size=self.kernel, dropout=self.dropout, normalization=self.normalization, activation=self.activation)
        self.Contraction_Block = []
        self.Expansion_Block = []
        self.Amplitude_Output = Amplitude_Output_Complex(
            b_skip=self.global_skip)
        self.Phase_Output = Phase_Output_Complex(b_skip=self.global_skip)
        self.Conv_Stack = Conv_Stack_Complex(
            kernel_size=self.kernel, dropout=self.dropout, normalization=self.normalization, activation=self.activation)
        for _ in range(self.depth):
            self.Contraction_Block.append(Contraction_Block_Complex(kernel_size=self.kernel, dropout=self.dropout,
                                                                    normalization=self.normalization, activation=self.activation))
            self.Expansion_Block.append(Expansion_Block_Complex(kernel_size=self.kernel, dropout=self.dropout,
                                                                normalization=self.normalization, activation=self.activation))
        self.Output_Wave = ComplexConv2D(
            1, [3, 3], padding='same', activation='linear')
        self.Deploy_Output = Deploy_Output()
        self.Cast_Input = Cast_Input()

    def build(self):
        super().build(input_shape=(
            None, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
        return self

    def call(self, inputs):

        x = self.Standardization_Layer(inputs)
        x, probe = self.Cast_Input(x)
        x = self.Convolution_Block(x)

        skips = list()
        for i in range(self.depth):
            x, ct = self.Contraction_Block[i](x)
            skips.append(ct)

        for i in range(self.depth):
            x = self.Expansion_Block[i](x, skips[-(i+1)])

        x = self.Output_Wave(x)
        x = x + probe
        x_a = self.Amplitude_Output(inputs, x)
        x_p = self.Phase_Output(inputs, x)
        x_o = tf.concat((x_a, x_p), -1)

        if self.deploy:
            return self.Deploy_Output(x_o)
        else:
            return x_o

    def summary(self):
        tfk.Model(inputs=self.Input, outputs=self.call(self.Input)).summary()

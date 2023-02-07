from tensorflow import keras as tfk
from ap_architectures.layers import *
import tensorflow as tf
if tuple(map(int, tf.__version__.split('.'))) >= (2,9,0):
    from ap_architectures.layers_complex import *


class UNET(tfk.Model):
    def __init__(self, prms, input_shape, deploy=False, probe=None, bs=None, **kwargs):
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
        self.global_cat = bool(prms['global_cat'])
        self._name = "UNET_" + str(self.filters) + "_D" + str(self.depth) + \
            "_N" + str(self.normalization) + "_A" + str(self.activation)
        self.probe = probe

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

        self.Input = kl.Input(
            shape=(self.x_shape[0], self.x_shape[1], self.x_shape[2]), name='cbeds')
        if deploy:
            self.Standardization_Layer = Standardization_Layer_Deployed(probe, bs)
            # self.Standardization_Layer = Standardization_Layer()
            self.probe = tf.tile(probe[tf.newaxis,...],[bs,1,1,1])
        else:
            self.Standardization_Layer = Standardization_Layer()
        
        # Initialization
        self.kernel_initializer = tfk.initializers.HeNormal(seed=1310)

        self.Convolution_Block = Convolution_Block(filters=self.filters, kernel_size=self.kernel, initializer=self.kernel_initializer, kernel_regularizer=self.w_regul,
                                                   activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation)
        self.Contraction_Block = []
        self.Expansion_Block = []
        
        for _ in range(self.depth):
            self.Contraction_Block.append(Contraction_Block(initializer=self.kernel_initializer, kernel_regularizer=self.w_regul,
                                          activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation))
            self.Expansion_Block.append(Expansion_Block(initializer=self.kernel_initializer, kernel_regularizer=self.w_regul,
                                        activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation))
        if deploy:
            self.Amplitude_Output = Amplitude_Output(b_skip=self.global_skip, b_cat=self.global_cat,beam=self.probe[...,0])
            self.Phase_Output = Phase_Output(b_skip=self.global_skip, b_cat=self.global_cat,beam=self.probe[...,1])
            self.Deploy_Output = Deploy_Output()
        else:
            self.Amplitude_Output = Amplitude_Output(b_skip=self.global_skip, b_cat=self.global_cat)
            self.Phase_Output = Phase_Output(b_skip=self.global_skip, b_cat=self.global_cat)

    def build(self, bs=None):
        super().build(input_shape=(
            bs, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
        return self

    def call(self, inputs, training=False):
        if self.deploy:
            bs = tf.shape(inputs)[0]
            inputs = tf.cast(inputs,tf.float32)
            probe = self.probe[:bs,...]
        if self.probe is None:
            probe = inputs[..., -2:]

        x = self.Standardization_Layer(inputs)

        x = self.Convolution_Block(x)
        skips = list()
        for i in range(self.depth):
            x, ct = self.Contraction_Block[i](x)
            skips.append(ct)

        for i in range(self.depth):
            x = self.Expansion_Block[i](x, skips[-(i+1)])

        x_a = self.Amplitude_Output(x,probe[...,0])
        x_p = self.Phase_Output(x,probe[...,1])
        
        x_o = tf.concat((x_a, x_p), -1)

        if self.deploy:
            return self.Deploy_Output(x_o, probe)
        else:
            return x_o

    # def summary(self):
    #     tfk.Model(inputs=self.Input, outputs=self.call(self.Input)).summary()


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
        self.global_cat = bool(prms['global_cat'])
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
        self.Amplitude_Output = Amplitude_Output(
            b_skip=self.global_skip)
        self.Phase_Output = Phase_Output(b_skip=self.global_skip)
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

    def build(self, bs=None):
        super().build(input_shape=(
            bs, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
        return self

    def call(self, inputs):
        if self.deploy:
            inputs = tf.cast(inputs,tf.float32)
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
        x_a = self.Amplitude_Output(inputs, tf.math.abs(x))
        x_p = self.Phase_Output(inputs, tf.math.angle(x))
        x_o = tf.concat((x_a, x_p), -1)

        if self.deploy:
            return self.Deploy_Output(inputs, x_o)
        else:
            return x_o

class TFLITE_Model:
    def __init__(self, model_path):
        self.model_path = model_path

        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Test the model on random input data.
        self.input_shape = self.input_details[0]['shape']

        
    def predict_on_batch(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[0]['index'])
        return result
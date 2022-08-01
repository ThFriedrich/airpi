from tensorflow import keras as tfk
from ap_architectures.layers import *

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
        # self.kernel_initializer = tfk.initializers.VarianceScaling(
        #     scale=1.0, mode='fan_in', distribution='truncated_normal', seed=1310)
        self.kernel_initializer = tfk.initializers.HeNormal(seed=1310)
        self.Input = kl.Input(shape=(self.x_shape[0], self.x_shape[1], self.x_shape[2]), name='cbeds')
        self.Standardization_Layer = Standardization_Layer()
        self.Convolution_Block = Convolution_Block(filters = self.filters, kernel_size=self.kernel, initializer=self.kernel_initializer, kernel_regularizer=self.w_regul, activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation)
        self.Contraction_Block = []
        self.Expansion_Block = []
        self.amplitude_output = amplitude_output(b_skip=self.global_skip)
        self.phase_output = phase_output()
        self.Conv_Stack = Conv_Stack(self.stack_n, kernel_size=self.kernel, initializer=self.kernel_initializer, kernel_regularizer=self.w_regul, activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation)
        for _ in range(self.depth-1):
            self.Contraction_Block.append(Contraction_Block(initializer=self.kernel_initializer, kernel_regularizer=self.w_regul, activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation))
            self.Expansion_Block.append(Expansion_Block(initializer=self.kernel_initializer, kernel_regularizer=self.w_regul, activity_regularizer=self.a_regul, dropout=self.dropout, normalization=self.normalization, activation=self.activation))
        self.deploy_output = Deploy_Output()

    def build(self):
        super().build(input_shape=(None, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
        return self
    
    def call(self, inputs, training=False):

        x = self.Standardization_Layer(inputs)
        x = self.Convolution_Block(x)
        
        skips = list()
        for i in range(self.depth-1):
            x, ct = self.Contraction_Block[i](x)
            skips.append(ct)
        
        x = self.Conv_Stack(x)
        
        for i in range(self.depth-1):
            x = self.Expansion_Block[i](x, skips[-(i+1)])

        x_a = self.amplitude_output(inputs, x)
        x_p = self.phase_output(x)
        x_o = tfk.layers.Concatenate(dtype=tf.float32)((x_a, x_p))

        if self.deploy:
            return self.deploy_output(x_o)
        else:
            return x_o
        

    def summary(self):
        tfk.Model(inputs=self.Input, outputs=self.call(self.Input)).summary()
from keras.layers import Input, Dense, LeakyReLU, Layer, Dropout
from keras.models import Model
import keras.backend as K
from keras import regularizers
from keras.losses import categorical_crossentropy, binary_crossentropy, hinge, squared_hinge
from keras import backend as K
from keras.layers import Layer
from keras.optimizers import Adam
from keras.initializers import *
from keras.callbacks import ModelCheckpoint

class LassoLayer(Layer):
    def __init__(self, output_dim, kernel_regularizer=None, **kwargs):        
        self.output_dim = output_dim
        self.kernel_regularizer = regularizers.get(kernel_regularizer)        
        super(LassoLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1, input_shape[1]),
                                      initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                                      trainable=True, 
                                      regularizer=self.kernel_regularizer)        
        super(LassoLayer, self).build(input_shape)  
        
    def call(self, x):
        return x * self.kernel
    
    def compute_output_shape(self, input_shape):
        return (None, input_shape[1])
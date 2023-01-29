import tensorflow as tf
from tensorflow import keras
from keras_cv.models.__internal__.darknet_utils import (
    DarknetConvBlock,
    ResidualBlocks,
)

class darknet53(keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.conv1 = DarknetConvBlock(filters=(3, 3), strides=1)
        self.conv2 = DarknetConvBlock(filters=(3, 3), strides=1)
        self.residual = ResidualBlocks(1)

class darknetblock(keras.layers.Layer):
    def __init__(
        self
    ):
        super().__init__()
        
        # layers instantialize
    
    def call(self, inputs, training=False):

        return  # outputs will be returned

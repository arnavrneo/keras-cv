import tensorflow as tf
from tensorflow import keras
from keras_cv.models.__internal__.darknet_utils import (
    DarknetConvBlock,
    ResidualBlocks,
)

# darknet53 body


class darknet53_body(keras.layers.Layer):
    ''' 52 conv2d layers '''
    def __init__(self):
        super().__init__()
        self.conv0 = DarknetConvBlock(filters=32, kernel_size=(3, 3), strides=1)
        self.res0 = ResidualBlocks(64, 1)
        self.res1 = ResidualBlocks(128, 2)
        self.res2 = ResidualBlocks(256, 8)
        self.res3 = ResidualBlocks(512, 8)
        self.res4 = ResidualBlocks(1024, 4)
    
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.res0(inputs)
        x = self.res1(inputs)
        x = self.res2(inputs)
        x = self.res3(inputs)
        x = self.res4(inputs)
        return x


class Yolov3Model(keras.layers.Layer):
    def __init__(
        self,
        in_channels=[256, 512, 1024],
        depth_multiplier=1.0,
        width_multiplier=1.0,
        activation="silu"
    ):
        super().__init__()
        self.in_channels = in_channels
        # layers instantialize

        self.conv0 = DarknetConvBlock(
            filters=int(in_channels[1] * width_multiplier),
            kernel_size=1,
            strides=1,
            activation=activation
        )
    
    def call(self, inputs, training=False):

        return  # outputs will be returned

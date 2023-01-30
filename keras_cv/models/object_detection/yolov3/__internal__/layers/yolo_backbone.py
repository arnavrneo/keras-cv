import tensorflow as tf
from tensorflow import keras
from keras_cv.models.__internal__.darknet_utils import (
    DarknetConvBlock,
    ResidualBlocks,
)

# darknet53 body


class Darknet53_body(keras.layers.Layer):
    ''' 52 conv2d layers '''
    def __init__(self):
        super().__init__()
        self.conv0 = DarknetConvBlock(filters=32, kernel_size=(3, 3), strides=2)
        self.conv1 = DarknetConvBlock(filters=64, kernel_size=(3, 3), strides=2)
        self.res0 = ResidualBlocks(32, 1)
        self.conv2 = DarknetConvBlock(filters=128, kernel_size=(3, 3), strides=2)
        self.res1 = ResidualBlocks(64, 2)
        self.conv3 = DarknetConvBlock(filters=256, kernel_size=(3, 3), strides=1)
        self.res2 = ResidualBlocks(128, 8)
        self.conv4 = DarknetConvBlock(filters=512, kernel_size=(3, 3), strides=1)
        self.res3 = ResidualBlocks(256, 8)
        self.conv5 = DarknetConvBlock(filters=1024, kernel_size=(3, 3), strides=1)
        self.res4 = ResidualBlocks(512, 4)
    
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.res0(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)

        route1 = x   # 128

        x = self.conv4(x)
        x = self.res3(x)

        route2 = x   # 256

        x = self.conv5(x)
        x = self.res4(x)

        route3 = x   # 512
        return route1, route2, route3


class Yolov3_block(keras.layers.Layer):
    def __init__(
        self,
        filters
    ):
        self.filters = filters
        super().__init__()
        self.conv0 = DarknetConvBlock(filters=filters, kernel_size=1, strides=1)
        self.conv1 = DarknetConvBlock(filters=2*filters, kernel_size=3, strides=1)
        self.conv2 = DarknetConvBlock(filters=filters, kernel_size=1, strides=1)
        self.conv3 = DarknetConvBlock(filters=2*filters, kernel_size=3, strides=1)
        self.conv4 = DarknetConvBlock(filters=filters, kernel_size=1, strides=1)
        self.conv5 = DarknetConvBlock(filters=2*filters, kernel_size=3, strides=1)
        
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        route = x

        x = self.conv5(x)
        return route, x


class Yolov3Model(keras.layers.Layer):
    def __init__(
        self,
        inputs,
        num_anchors,
        img_size

    ):
        super().__init__()
        # layers instantialize
        self.res0 = ResidualBlocks(1024, 2)
        
    
    def call(self, inputs, training=False):

        return  # outputs will be returned

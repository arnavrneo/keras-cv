import tensorflow as tf
from tensorflow import keras
from keras_cv.models.__internal__.darknet_utils import (
    DarknetConvBlock,
    ResidualBlocks,
)

# darknet53 body


class Darknet53(keras.layers.Layer):
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


class Upsample(keras.layers.Layer):
    @classmethod
    def call(inputs, out_shape, data_format):

        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
            new_height = out_shape[3]
            new_width = out_shape[2]
        else:
            new_height = out_shape[2]
            new_width = out_shape[1]

        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        return inputs

# class Yolov3Model(keras.layers.Layer):
#     def __init__(
#         self,
#         anchors,
#         n_classes,
#         img_size,
#         data_format

#     ):
#         super().__init__()
#         # layers instantialize
#         self.anchors = anchors
#         self.n_anchors = len(self.anchors)
#         self.n_classes = n_classes
#         self.data_format = data_format
#         self.img_size = img_size
        
#     def call(self, inputs, training=False):

#         inputs = keras.layers.Conv2D(
#                                 inputs,
#                                 filters=self.n_anchors * (5 + self.n_classes),
#                                 kernel_size=1,
#                                 strides=1,
#                                 user_bias=True,
#                                 data_format=self.data_format
#                             )
#         shape = inputs.get_shape().as_list()
#         grid_shape = shape[2:4] if self.data_format == "channels_first" else shape[1:3]

#         if self.data_format == "channels_first":
#             inputs = tf.transpose(inputs, [0, 2, 3, 1])
        
#         inputs = tf.reshape(inputs, [-1, self.n_anchors * grid_shape[0] * grid_shape[1], 5 + self.n_classes])

#         strides = (self.img_size[0] // grid_shape[0], self.img_size[1] // grid_shape[1])

#         box_centers, box_shapes, confidence, classes = \
#             tf.split(inputs, [2, 2, 1, self.n_classes], axis=-1)

#         x = tf.range(grid_shape[0], dtype=tf.float32)
#         y = tf.range(grid_shape[1], dtype=tf.float32)
#         x_offset, y_offset = tf.meshgrid(x, y)
#         x_offset = tf.reshape(x_offset, (-1, 1))
#         y_offset = tf.reshape(y_offset, (-1, 1))
#         x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
#         x_y_offset = tf.tile(x_y_offset, [1, self.n_anchors])
#         x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
#         box_centers = tf.nn.sigmoid(box_centers)
#         box_centers = (box_centers + x_y_offset) * strides

#         anchors = tf.tile(self.anchors, [grid_shape[0] * grid_shape[1], 1])
#         box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)

#         confidence = tf.nn.sigmoid(confidence)

#         classes = tf.nn.sigmoid(classes)

#         inputs = tf.concat([box_centers, box_shapes,
#                             confidence, classes], axis=-1)

#         return inputs

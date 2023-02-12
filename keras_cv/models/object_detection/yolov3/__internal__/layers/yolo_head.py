import tensorflow as tf
from tensorflow import keras
from keras_cv.models.__internal__.darknet_utils import (
    DarknetConvBlock
)

class Yolohead(keras.layers.Layer):

    def __init__(
        self,
        anchors,
        n_classes,
        img_size,
        data_format

    ):
        super().__init__()
        # layers instantialize
        self.anchors = anchors
        self.n_anchors = len(self.anchors)
        self.n_classes = n_classes
        self.data_format = data_format
        self.img_size = img_size
        
    def call(self, inputs, training=False):

        inputs = keras.layers.Conv2D(
                                inputs,
                                filters=self.n_anchors * (5 + self.n_classes),
                                kernel_size=1,
                                strides=1,
                                user_bias=True,
                                data_format=self.data_format
                            )
        shape = inputs.get_shape().as_list()
        grid_shape = shape[2:4] if self.data_format == "channels_first" else shape[1:3]

        if self.data_format == "channels_first":
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        
        inputs = tf.reshape(inputs, [-1, self.n_anchors * grid_shape[0] * grid_shape[1], 5 + self.n_classes])

        strides = (self.img_size[0] // grid_shape[0], self.img_size[1] // grid_shape[1])

        box_centers, box_shapes, confidence, classes = \
            tf.split(inputs, [2, 2, 1, self.n_classes], axis=-1)

        x = tf.range(grid_shape[0], dtype=tf.float32)
        y = tf.range(grid_shape[1], dtype=tf.float32)
        x_offset, y_offset = tf.meshgrid(x, y)
        x_offset = tf.reshape(x_offset, (-1, 1))
        y_offset = tf.reshape(y_offset, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.tile(x_y_offset, [1, self.n_anchors])
        x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
        box_centers = tf.nn.sigmoid(box_centers)
        box_centers = (box_centers + x_y_offset) * strides

        anchors = tf.tile(self.anchors, [grid_shape[0] * grid_shape[1], 1])
        box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)

        confidence = tf.nn.sigmoid(confidence)

        classes = tf.nn.sigmoid(classes)

        inputs = tf.concat([box_centers, box_shapes,
                            confidence, classes], axis=-1)

        return inputs

        




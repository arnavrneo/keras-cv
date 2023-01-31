import tensorflow as tf
from tensorflow import keras
from keras_cv.models.__internal__.darknet_utils import (
    DarknetConvBlock
)


class Yolohead(keras.layers.Layer):

    def __init__(
        self,
        n_classes,
        anchors,
        img_size,
        img_format
    ):
        super().__init__()
        self.n_classes = n_classes
        self.anchors = anchors
        self.img_size = img_size
        self.img_format = img_format 
        self.n_anchors = len(self.anchors)

        self.conv = DarknetConvBlock(
            filters=self.n_anchors,
            kernel_size=1,
            strides=1
        ) 
        
    def call(self, inputs):
        a = self.conv(inputs)
        shape = a.TensorShape().as_list()
        grid_shape = self.shape[2:4] if self.img_format == "channels_first" else shape[1:3]
        if self.img_format == "channels_first":
            a = tf.transpose(a, [0, 2, 3, 1])
        a = tf.reshape(a, [-1, self.n_anchors*grid_shape[0]*grid_shape[1]*grid_shape[1], 5 + self.n_classes])
        strides = (self.img_size[0]) // grid_shape[0], self.img_size[1] // grid_shape[1]

        bbox_centres, bbox_shapes, conf, classes = tf.split(inputs, [2, 2, 1, self.n_classes], axis=-1)

        x = tf.range(grid_shape[0], dtype=tf.float32)
        y = tf.range(grid_shape[1], dtype=tf.float32)
        x_offset, y_offset = tf.meshgrid(x, y)
        x_offset = tf.reshape(x_offset, (-1, 1))
        y_offset = tf.reshape(y_offset, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.tile(x_y_offset, [1, self.n_anchors])
        x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
        bbox_centres = tf.nn.sigmoid(bbox_centres)
        bbox_centres = (bbox_centres + x_y_offset) * strides

        self.anchors = tf.tile(self.anchors, [grid_shape[0] * grid_shape[1], 1])
        bbox_shapes = tf.exp(bbox_shapes) * tf.to_float(self.anchors)

        conf = tf.nn.sigmoid(conf)

        classes = tf.nn.sigmoid(classes)

        inputs = tf.concat([bbox_centres, bbox_shapes,
                            conf, classes], axis=-1)

        return inputs
        
        




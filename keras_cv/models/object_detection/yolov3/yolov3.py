import keras_cv
from keras_cv import bounding_box
import numpy as np
import tensorflow as tf
from keras_cv import layers as cv_layers
from keras_cv.models.__internal__.darknet_utils import (
    DarknetConvBlock,
    ResidualBlocks,
)
from __internal__.layers.yolo_backbone import Darknet53, Yolov3_block, Upsample
from __internal__.layers.yolo_head import Yolohead
from __internal__.layers.common import ANCHORS
# it will work 
# CONTINUE THE WORK. YOU NEED TO DO IT PLEASE

layer = cv_layers.MultiClassNonMaxSuppression(
    bounding_box_format='xyxy',
    from_logits=True,
    max_detections=100,
)


class YoloV3:

    def __init__(
        self,
        classes,
        bounding_box_format,
        img_size,
        output_size,
        iou_threshold,
        confidence_threshold,
        data_format=None
    ):
        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'

        self.classes = classes
        self.bounding_box_format = bounding_box_format
        self.img_size = img_size
        self.output_size = output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

    def call(self, inputs, training):
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        
        inputs = inputs/255
        # CALL MODELS BELOW
        route1, route2, inputs = Darknet53()(inputs) 

        # FOR 512
        route, inputs = Yolov3_block(filters=512)(inputs)

        # DETECTIONS
        detect1 = Yolohead(
            inputs,
            n_classes=self.classes,
            anchors=ANCHORS[6:9],
            img_size=self.img_size,
            data_format=self.data_format
        )

        inputs = DarknetConvBlock(
            filters=256,
            kernel_size=1
        )(route)

        upsample_size = route2.get_shape().as_list()
        inputs = Upsample()(
            inputs, out_shape=upsample_size, 
            data_format=self.data_format)

        axis = 1 if self.data_format == 'channels_first' else 3

        inputs = tf.concat([inputs, route2], axis=axis)

        # FOR 256
        route, inputs = Yolov3_block(filters=512)(inputs)

        # DETECTIONS
        detect2 = Yolohead(
            inputs,
            n_classes=self.classes,
            anchors=ANCHORS[3:6],
            img_size=self.img_size,
            data_format=self.data_format
        )

        inputs = DarknetConvBlock(
            filters=128,
            kernel_size=1
        )(route)

        upsample_size = route1.get_shape().as_list()
        inputs = Upsample()(
            inputs, out_shape=upsample_size,
            data_format=self.data_format)

        axis = 1 if self.data_format == 'channels_first' else 3

        inputs = tf.concat([inputs, route1], axis=axis)

        # FOR 128
        route, inputs = Yolov3_block(filters=128)(inputs)

        # DETECTIONS
        detect3 = Yolohead(
            inputs,
            n_classes=self.classes,
            anchors=ANCHORS[0:3],
            img_size=self.img_size,
            data_format=self.data_format
        )

        inputs = tf.concat([detect1, detect2, detect3], axis=1)

        # BUILDING BOXES AND DOING NMS (LAST STEP)
        # inputs = build_boxes(inputs)
        # boxes_dicts = NMS()

        # return boxes_dicts

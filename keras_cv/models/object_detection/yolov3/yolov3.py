import keras_cv
from keras_cv import bounding_box
import numpy as np
import tensorflow as tf
from keras_cv import layers as cv_layers
from __internal__.layers import *

# it works 
# CONTINUE THE WORK. YOU NEED TO DO IT PLEASE

layer = cv_layers.MultiClassNonMaxSuppression(
    bounding_box_format='xyxy',
    from_logits=True,
    max_detections=100,
)

layer

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
# CALL DARKNET MODEL BELOW
#    route1, route2, inputs =     

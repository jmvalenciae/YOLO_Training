from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.yolov4 import (
    YoloV4, YoloLoss,
    yolov4_anchors, yolov4_anchor_masks, xyscales, strides
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

flags.DEFINE_string('dataset', '/home/jorge/Desktop/yolov3-tf2-master/data/MOT20_tfrecords/MOT20_train_00000-of-00003_00000-of-00002.records', 'path to dataset')
flags.DEFINE_string('val_dataset', '/home/jorge/Desktop/yolov3-tf2-master/data/MOT20_tfrecords/MOT20_val_00000-of-00003_00002-of-00002.records', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', '/home/jorge/Desktop/yolov3-tf2-master/checkpoints/yolov4.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'darknet',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 608, 'image size')
flags.DEFINE_integer('epochs', 1, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', 80, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')


def main(_argv):
    model = YoloV4(FLAGS.size, training=True, classes=FLAGS.num_classes)
    anchors = yolov4_anchors
    anchor_masks = yolov4_anchor_masks

    train_dataset = dataset.load_fake_dataset()
    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    tsteps = sum(1 for _ in train_dataset)
    train_dataset = train_dataset.shuffle(buffer_size=256, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=True)
    train_dataset = train_dataset.repeat(FLAGS.epochs)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets_yolov4(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_fake_dataset()
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    vsteps = sum(1 for _ in val_dataset)
    val_dataset = val_dataset.batch(FLAGS.batch_size, drop_remainder=True)
    val_dataset = val_dataset.repeat(FLAGS.epochs)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets_yolov4(y, anchors, anchor_masks, FLAGS.size)))

    print("this is a test")
    print("train_dataset",train_dataset)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import math
import sys
import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import freeze_graph

import models.data as data
import models.select as select

from train import get_run_dir

import ngraph_bridge

FLAGS = tf.app.flags.FLAGS

def export_model():
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        IMAGE_SIZE = 24 if FLAGS.data_aug else 32

        images = tf.constant(
            1,
            dtype=tf.float32,
            shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3]
        )

        model = select.by_name(FLAGS.model, training=False)
        logits = model.inference(images)

        print('nodes', [n.name for n in tf.get_default_graph().as_graph_def().node])

        output_graph_def = optimize_for_inference_lib.optimize_for_inference(tf.get_default_graph().as_graph_def(),
                  ['input'], ['output'], dtypes.float32.as_datatype_enum, False)



        # Build a Graph that computes the logits predictions from the
        # inference model.
        print("loaded saved graph")
        with tf.Session() as sess:
            sess.run(logits, feed_dict = {images: np.random.random((1, IMAGE_SIZE, IMAGE_SIZE, 3))})
        print('done with session')

def main(argv=None):
  export_model()


if __name__ == '__main__':
  tf.app.run()


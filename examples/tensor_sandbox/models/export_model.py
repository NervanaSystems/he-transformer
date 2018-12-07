
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import math
import sys
import time

import numpy as np
import tensorflow as tf

import models.data as data
import models.select as select
import os

from train import get_run_dir

import ngraph_bridge

FLAGS = None

import models.data as data
import models.select as select

FLAGS = tf.app.flags.FLAGS

def export_model():
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.

        images = tf.constant(
            1,
            dtype=tf.float32,
            shape=[1, 24, 24, 3]
        )

        model = select.by_name(FLAGS.model, bool_training=True)
        logits = model.inference(images)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        print("loaded saved graph")
        with tf.Session() as sess:
            sess.run(logits, feed_dict = {images: np.random.random((1, 24, 24, 3))})
        print('done with session')

def main(argv=None):
  export_model()


if __name__ == '__main__':
  tf.app.run()



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

FLAGS = None

import models.data as data
import models.select as select

FLAGS = tf.app.flags.FLAGS

def save_weights():
  """Saves CIFAR10 weights"""
  FLAGS.resume = True # Get saved weights, not new ones
  run_dir = get_run_dir(FLAGS.log_dir, FLAGS.model)
  checkpoint_dir = os.path.join(run_dir, 'train')

  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    print('data dir', FLAGS.data_dir)
    images, labels = data.train_inputs(data_dir=FLAGS.data_dir)

    model = select.by_name(FLAGS.model, bool_training=True)

    print('FLAGS.model', FLAGS.model)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images)

    # Restore the moving average version of the learned variables for eval.
    variables_averages = tf.train.ExponentialMovingAverage(1.0) # 1.0 decay is unused
    variables_to_restore = variables_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
      print('Creating session')

      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        print('ckpt_dir', checkpoint_dir)
        print('ckpt.model_checkpoint_path', ckpt.model_checkpoint_path)
        print('ckpt', ckpt)
        return

      # Save variables
      for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        weight = (sess.run([var]))[0].flatten().tolist()
        filename = model.name_to_filename(var.name)
        dir_name = filename.rsplit('/',1)[0]
        print('dirnmame', dir_name)
        os.makedirs(dir_name, exist_ok=True)

        print("saving", filename)
        np.savetxt(str(filename), weight)

def main(argv=None):
  data.maybe_download_and_extract(FLAGS.data_dir)
  save_weights()


if __name__ == '__main__':
    tf.app.run()

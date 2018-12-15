
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
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import freeze_graph
from tensorflow.python.platform import gfile

import ngraph_bridge

from train import get_run_dir

FLAGS = tf.app.flags.FLAGS

def save_weights():
  """Saves CIFAR10 weights"""
  FLAGS.resume = True # Get saved weights, not new ones
  run_dir = get_run_dir(FLAGS.log_dir, FLAGS.model)
  checkpoint_dir = os.path.join(run_dir, 'train')
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  train_graph = os.path.join(checkpoint_dir, 'graph.pbtxt')
  frozen_graph = os.path.join(checkpoint_dir, 'graph_constants.pb')

  with tf.Session() as sess:
    # TODO this should be a placeholder, right?
	# Build a new inference graph, with variables to be restored from
	# training graph.
    IMAGE_SIZE = 24 if FLAGS.data_aug else 32
    images = tf.constant(
        1,
        dtype=tf.float32,
        shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3]
    )

    model = select.by_name(FLAGS.model, training=False)
    images  = tf.identity(images, 'XXX')
    logits = model.inference(images)
    logits = tf.identity(logits, 'YYY')

	# TODO we want to find the exponential mean/var computed during training
	# and hook that up in place of the last batch mean/var in the inference graph
	# but we don't want to create a new/uninitialized variable right?

	# Restore values from the trained model into corresponding variables in the
	# inference graph.
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    assert ckpt and ckpt.model_checkpoint_path, "No checkpoint found in {}".format(checkpoint_dir)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Write fully-assembled inference graph to a file, so freeze_graph can use it
    tf.io.write_graph(sess.graph, checkpoint_dir, 'inference_graph.pbtxt', as_text=True)

    # Freeze graph, converting variables to inline-constants in pb file
    constant_graph = os.path.join(checkpoint_dir, 'graph_constants.pb')
    freeze_graph.freeze_graph(
        input_graph=os.path.join(checkpoint_dir, 'inference_graph.pbtxt'),
        input_saver="",
        input_binary=False,
        input_checkpoint=ckpt.model_checkpoint_path,
        output_node_names='YYY',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        initializer_nodes=[],
        output_graph=os.path.join(checkpoint_dir, 'graph_constants.pb'),
        clear_devices=True)


	# Load frozen graph into a graph_def for optimize_lib to use
    with gfile.FastGFile(constant_graph,'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')

    # Optimize graph for inference, folding Batch Norm ops into conv/MM
    fused_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def=graph_def,
      input_node_names=['XXX'],
      output_node_names=['YYY'],
      placeholder_type_enum=dtypes.float32.as_datatype_enum,
      toco_compatible=False)

    print('Optimized for inference.')

    tf.io.write_graph(fused_graph_def, checkpoint_dir, name='fused_graph.pb', as_text=False)


def serialize_model():
  print('Serializing model')

  FLAGS.resume = True # Get saved weights, not new ones
  run_dir = get_run_dir(FLAGS.log_dir, FLAGS.model)
  checkpoint_dir = os.path.join(run_dir, 'train')
  fused_graph_file = os.path.join(checkpoint_dir, 'fused_graph.pb')
  print('fused_graph_file', fused_graph_file)

  IMAGE_SIZE = 24 if FLAGS.data_aug else 32

  #with tf.Session() as sess:

  with gfile.FastGFile(fused_graph_file,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    assert len(graph.get_operations()) == 0, "Assuming an empty graph here to populate with fused graph"
    tf.import_graph_def(graph_def, name='')

  print('nodes', [n.name for n in graph_def.node])
  XXX = graph.get_tensor_by_name('XXX:0')
  YYY = graph.get_tensor_by_name('YYY:0')

  print("Serializing model")
  with tf.Session(graph=graph) as sess:
    sess.run(YYY, feed_dict = {XXX: np.random.random((1, IMAGE_SIZE, IMAGE_SIZE, 3))})


def main(argv=None):
  data.maybe_download_and_extract(FLAGS.data_dir)
  save_weights()

  serialize_model()


if __name__ == '__main__':
    tf.app.run()


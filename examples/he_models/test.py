
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

tf.app.flags.DEFINE_bool('report_accuracy', True, "Whether or not to report accuracy")

def save_weights():
  """Saves CIFAR10 weights"""
  FLAGS.resume = True # Get saved weights, not new ones
  run_dir = get_run_dir(FLAGS.log_dir, FLAGS.model)
  checkpoint_dir = os.path.join(run_dir, 'train')

  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    print('data dir', FLAGS.data_dir)
    images, labels = data.train_inputs(data_dir=FLAGS.data_dir)
    model = select.by_name(FLAGS.model, training=True)

    print('FLAGS.model', FLAGS.model)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images)
    saver = tf.train.Saver()

    # TODO: use saved averages?

    with tf.Session() as sess:
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
        os.makedirs(dir_name, exist_ok=True)

        print("saving", filename)
        np.savetxt(str(filename), weight)

def optimize_model_for_inference():
  """Saves CIFAR10 weights"""
  FLAGS.resume = True # Get saved weights, not new ones
  run_dir = get_run_dir(FLAGS.log_dir, FLAGS.model)
  checkpoint_dir = os.path.join(run_dir, 'train')
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  train_graph = os.path.join(checkpoint_dir, 'graph.pbtxt')
  frozen_graph = os.path.join(checkpoint_dir, 'graph_constants.pb')
  fused_graph = os.path.join(checkpoint_dir, 'fused_graph.pb')
  if os.path.isfile(fused_graph):
    print("Graph already optimized")
    return

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

def report_accuracy(logits, labels):
  print("predictions", np.argmax(logits, 1), 'labels', labels)
  correct_prediction = np.equal(np.argmax(logits, 1), labels)
  error_count = np.size(correct_prediction) - np.sum(correct_prediction)
  test_accuracy = np.mean(correct_prediction)

  print('Error count', error_count, 'of', len(labels), 'elements.')
  print('Accuracy ', test_accuracy)


def serialize_model():
  print('Serializing model')

  FLAGS.resume = True # Get saved weights, not new ones
  run_dir = get_run_dir(FLAGS.log_dir, FLAGS.model)
  checkpoint_dir = os.path.join(run_dir, 'train')
  fused_graph_file = os.path.join(checkpoint_dir, 'fused_graph.pb')
  print('fused_graph_file', fused_graph_file)

  eval_data, eval_labels = data.numpy_eval_inputs(True, FLAGS.data_dir, FLAGS.batch_size)

  with gfile.FastGFile(fused_graph_file,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    assert len(graph.get_operations()) == 0, "Assuming an empty graph here to populate with fused graph"
    tf.import_graph_def(graph_def, name='')

  print('nodes', [n.name for n in graph_def.node])
  XXX = graph.get_tensor_by_name('XXX:0')
  YYY = graph.get_tensor_by_name('YYY:0')

  print("Running model")
  with tf.Session(graph=graph) as sess:
    eval_batch_data = eval_data[0]
    eval_batch_label = eval_labels[0]

    YYY = sess.run(YYY, feed_dict = {XXX: eval_batch_data})

    if FLAGS.report_accuracy:
      report_accuracy(YYY, eval_batch_label)


def main(argv=None):
  data.maybe_download_and_extract(FLAGS.data_dir)

  save_weights()
  optimize_model_for_inference()

  serialize_model()


if __name__ == '__main__':
    tf.app.run()


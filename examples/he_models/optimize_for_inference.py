
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

  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    print('data dir', FLAGS.data_dir)
    #images, labels = data.train_inputs(data_dir=FLAGS.data_dir)

    IMAGE_SIZE = 24 if FLAGS.data_aug else 32

    images = tf.constant(
        1,
        dtype=tf.float32,
        shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3]
    )

    model = select.by_name(FLAGS.model, training=False)

    print('FLAGS.model', FLAGS.model)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    images  = tf.identity(images, 'XXX')
    logits = model.inference(images)
    logits = tf.identity(logits, 'YYY')

    # Restore the moving average version of the learned variables for eval.
    variables_averages = tf.train.ExponentialMovingAverage(1.0) # 1.0 decay is unused
    variables_to_restore = variables_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

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
      if False:
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
          weight = (sess.run([var]))[0].flatten().tolist()
          filename = model.name_to_filename(var.name)
          dir_name = filename.rsplit('/',1)[0]
          os.makedirs(dir_name, exist_ok=True)

          print("saving", filename)
          np.savetxt(str(filename), weight)

      print('nodes', [n.name for n in tf.get_default_graph().as_graph_def().node])
      print('input graph', os.path.join(checkpoint_dir, 'graph.pbtxt'))

      # Save graph, with variables as consatnts??
      constant_graph = os.path.join(checkpoint_dir, 'graph_constants.pb')
      freeze_graph.freeze_graph(
          input_graph=os.path.join(checkpoint_dir, 'graph.pbtxt'),
          input_saver="",
          input_binary=False,
          input_checkpoint=ckpt.model_checkpoint_path,
          output_node_names='YYY',
          restore_op_name='save/restore_all',
          filename_tensor_name='save/Const:0',
          initializer_nodes=[],
          output_graph=os.path.join(checkpoint_dir, 'graph_constants.pb'),
          clear_devices=True)

      print('\nfroze graph!\n')

      print('Optimizing for inference')
      print('input graph at', os.path.join(checkpoint_dir, 'graph_constants.pb'))

      with gfile.FastGFile(constant_graph,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

      #print("graph_def", graph_def)

      # Load saved graph with variables as constants, and apply batch norm fusions
      fused_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def=graph_def,
        input_node_names=['XXX'],
        output_node_names=['YYY'],
        placeholder_type_enum=dtypes.float32.as_datatype_enum,
        toco_compatible=False)

      print('Optimized for inference.')

      #print('Output graph def', output_graph_def)

      tf.io.write_graph(fused_graph_def, checkpoint_dir, name='fused_graph.pb', as_text=False)

      #print('nodes', [n.name for n in tf.get_default_graph().as_graph_def().node])

      print('Saved fused graph\n\n')


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
    tf.import_graph_def(graph_def, name='')

  print('nodes', [n.name for n in graph_def.node])
  #print('nodes', [n for n in graph_def.node])


  XXX = graph.get_tensor_by_name('XXX:0')
  YYY = graph.get_tensor_by_name('YYY:0')

  with tf.Session(graph=graph) as sess:
    sess.run(YYY, feed_dict = {XXX: np.random.random((1, IMAGE_SIZE, IMAGE_SIZE, 3))})

    print('YYY', YYY)


def main(argv=None):
  data.maybe_download_and_extract(FLAGS.data_dir)
  save_weights()

  serialize_model()


if __name__ == '__main__':
    tf.app.run()



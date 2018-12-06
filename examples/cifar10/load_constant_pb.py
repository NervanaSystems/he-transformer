import ngraph_bridge

import tensorflow as tf

import cifar10

import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format


import time

def pbtxt_to_graphdef(infile, outpath, outfile):
  with open(infile, 'r') as f:
    graph_def = tf.GraphDef()
    file_content = f.read()
    text_format.Merge(file_content, graph_def)
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, outpath, outfile, as_text=False)

pbtxt_to_graphdef('./freeze/cifar10_const.pbtxt', './freeze/', 'cifar10_const.pb')

def load_model():
  GRAPH_PB_PATH = './freeze/cifar10_const.pb'

  with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()


    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    names = []
    for t in graph_nodes:
        names.append(t.name)
    print(names)

    #placeholders = [ op for op in graph_def.get_operations() if op.type == "Placeholder"]
    #print('placeholders', placeholders)

    #print('gd', graph_def)

  with tf.Graph().as_default() as g:
    with tf.Session() as sess:
      # Get images and labels for CIFAR-10.
      eval_data = 'test'
      images, labels = cifar10.inputs(eval_data=eval_data)

      images = images[0:10]
      labels = labels[0:10]

      # Build a Graph that computes the logits predictions from the
      # inference model.
      logits = cifar10.he_inference(images)

      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

        num_iter = 1 #int(math.ceil(FLAGS.num_examples / 128))
        #true_count = 0  # Counts the number of correct predictions.
        #total_sample_count = num_iter * 128
        step = 0
        while step < num_iter and not coord.should_stop():
          predictions = sess.run([logits])
          print('Sleeping')
          time.sleep(100)
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)
        print("Error in starting queue runners")

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

      print("done computing precision")


load_model()
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

# pbtxt_to_graphdef('./freeze/cifar10_const.pbtxt', './freeze/', 'cifar10_const.pb')

def load_model():
  GRAPH_PB_PATH = './freeze/cifar10_const.pbtxt'

  tf.reset_default_graph()
  with tf.Session() as sess:
    with open(GRAPH_PB_PATH, 'r') as f:
      graph_def = tf.GraphDef()
      file_content = f.read()
      text_format.Merge(file_content, graph_def)
      tf.import_graph_def(graph_def, name='')

    print("loaded graph")
    #print('graph_def', graph_def)
    # Get images and labels for CIFAR-10.
    g = tf.get_default_graph()

    print('loaded_graph', g)

    print('nodes', [n.name for n in g.as_graph_def().node])

    print('ops', [op.name for op in g.get_operations()])

    placeholders = [ op for op in g.get_operations() if op.type == "Placeholder"]
    variables = [ op for op in g.get_operations() if op.type == "Variable"]
    parameters = [ op for op in g.get_operations() if op.type == "Parameter"]

    print('parameters', parameters)
    print('placeholders', placeholders)
    print('variables', variables)

    eval_data = 'test'

    print([n.name for n in tf.get_default_graph().as_graph_def().node])

    final_op = [ op for op in g.get_operations() if op.name == "avg_pool/avg_pool"]
    final_op = final_op[0]

    print(final_op)

    # Run the session
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                        start=True))

      num_iter =1 # int(math.ceil(FLAGS.num_examples / 128))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * 128
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([final_op])
        print('Sleeping')
        time.sleep(100)
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count


      print('%s: precision @ 1 = %.3f' % (datetime.datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
      print("Error in starting queue runners")

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    print("done computing precision")


    # Exit

    exit(1)
    #images, labels = cifar10.inputs(eval_data=eval_data)

    images = images[0:10]
    labels = labels[0:10]


    # Build a Graph that computes the logits predictions from the
    # inference model.
    #logits = cifar10.he_inference(images)

    #print('nodes', [n.name for n in tf.get_default_graph().as_graph_def().node])

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
        predictions = sess.run('avg_pool/avg_pool')
        print('Sleeping')
        time.sleep(100)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
      print("Error in starting queue runners")

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    print("done computing precision")


load_model()
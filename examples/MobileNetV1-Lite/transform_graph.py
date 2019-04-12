import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.platform import gfile


def vars_to_constants():

    sess = tf.Session()

    orig_def = read_pb_file('./model/squeezenet.pb')

    # Extract subgraph clone0
    nodes_to_save = ['clone_0/squeezenet_v11/logits']
    graphdef_const = tf.graph_util.convert_variables_to_constants(
        sess, orig_def, nodes_to_save)

    tf.train.write_graph(
        graphdef_const, './model/', 'vars_to_constants.pb', as_text=False)

    # Remove training nodes
    #graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(
    #    sess.graph_def)

    #nodes = [n.name for n in graphdef_frozen.node]
    #print('nodes', len(nodes))


def read_pb_file(filename):
    sess = tf.Session()
    print("load graph", filename)
    with gfile.GFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    return graph_def


def read_nodes(filename, print_nodes=False):
    graph_def = read_pb_file(filename)
    nodes = [n.name for n in graph_def.node]
    print('nodes', len(nodes))

    if print_nodes:
        for node in sorted(nodes):
            print(node)


if __name__ == "__main__":
    #vars_to_constants()

    read_nodes('./model/mobilenet_v1_0.25_128_frozen.pb', True)
    #read_nodes('./model/vars_to_constants.pb')
    #read_nodes('./model/opt1.pb', True)

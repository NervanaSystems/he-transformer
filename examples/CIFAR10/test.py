import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.datasets import cifar10
import numpy as np

import ngraph_bridge
print(ngraph_bridge.get_currently_set_backend_name())


def parse_graph():
    f = gfile.FastGFile("./model/tf_model.pb", 'rb')
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()
    return graph_def


def run_inference():
    sess = tf.Session()

    sess.graph.as_default()
    # Import a serialized TensorFlow `GraphDef` protocol buffer
    # and place into the current default `Graph`.
    tf.import_graph_def(parse_graph())

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    output_tensor = sess.graph.get_tensor_by_name('import/dense_2/BiasAdd:0')
    input_tensor = sess.graph.get_tensor_by_name('import/input_1:0')
    y_pred = sess.run(output_tensor, {input_tensor: x_test})

    y_pred = np.argmax(y_pred, axis=1)
    y_test = y_test.flatten()

    accuracy = np.mean(y_pred == y_test)

    print('accuracy', accuracy)


if __name__ == "__main__":
    run_inference()

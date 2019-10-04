from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2
import tensorflow as tf
import ngraph_bridge
from keras import backend as K
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def client_config_from_flags(FLAGS, tensor_param_name):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = (
        rewriter_config_pb2.RewriterConfig.ONE)
    rewriter_options.min_graph_nodes = -1
    client_config = rewriter_options.custom_optimizers.add()
    client_config.name = "ngraph-optimizer"
    client_config.parameter_map["ngraph_backend"].s = FLAGS.backend.encode()
    client_config.parameter_map["device_id"].s = b''
    client_config.parameter_map[
        "encryption_parameters"].s = FLAGS.encryption_parameters.encode()
    client_config.parameter_map['enable_client'].s = (str(
        FLAGS.enable_client)).encode()
    if FLAGS.enable_client:
        client_config.parameter_map[tensor_param_name].s = b'client_input'

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_options)))

    return config


def main(FLAGS):
    config = client_config_from_flags(FLAGS, '')
    config = ngraph_bridge.update_config(config)
    sess = tf.Session(config=config)
    set_session(sess)

    model = DenseNet121(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000)

    img_path = 'grace_hopper.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--enable_client',
        type=str2bool,
        default=False,
        help='Enable the client')
    parser.add_argument(
        '--backend',
        type=str,
        default='HE_SEAL',
        help='Name of backend to use')
    parser.add_argument(
        '--encryption_parameters',
        type=str,
        default='',
        help=
        'Filename containing json description of encryption parameters, or json description itself'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)

import tensorflow as tf
import models.cnn as cnn
import models.deeper_cnn as deeper_cnn

tf.app.flags.DEFINE_string('model', 'cnn', """One of [cnn].""")
tf.app.flags.DEFINE_bool('train_poly_act', False, """True or False""")
tf.app.flags.DEFINE_bool('batch_norm', False, "True or False")


def by_name(name, FLAGS, training=False):
    name = name.lower()
    if name == 'cnn':
        print('train poly?', FLAGS.train_poly_act)
        print("bn?", FLAGS.batch_norm)
        model = cnn.CNN(
            training=training,
            train_poly_act=FLAGS.train_poly_act,
            batch_norm=FLAGS.batch_norm)
    elif name == 'deeper_cnn':
        print('train poly?', FLAGS.train_poly_act)
        print("bn?", FLAGS.batch_norm)
        model = deeper_cnn.DeeperCNN(
            training=training,
            train_poly_act=FLAGS.train_poly_act,
            batch_norm=FLAGS.batch_norm)
    else:
        raise ValueError('No such model %s' % name)
    return model

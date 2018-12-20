import tensorflow as tf
import models.cnn as cnn

tf.app.flags.DEFINE_string('model', 'cnn', """One of [cnn].""")


def by_name(name, training=False):
    name = name.lower()
    if name == 'cnn':
        model = cnn.CNN(training=training)
    else:
        raise ValueError('No such model %s' % name)
    return model

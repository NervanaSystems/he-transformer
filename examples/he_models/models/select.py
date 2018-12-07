import tensorflow as tf

import models.tutorial.tutorial as tutorial

tf.app.flags.DEFINE_string('model', 'tutorial',
        """One of [tuto].""")

def by_name(name, bool_training=False):
    if name == 'tutorial':
        model = tutorial.Tutorial(bool_training=bool_training)
    else:
        raise ValueError('No such model %s' % name)
    return model
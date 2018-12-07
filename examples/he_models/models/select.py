import tensorflow as tf

import models.cryptodl.cryptodl as cryptodl

tf.app.flags.DEFINE_string('model', 'CryptoDL',
        """One of [CryptoDL].""")

def by_name(name, bool_training=False):
    if name == 'CryptoDL':
        model = cryptodl.CryptoDL(bool_training=bool_training)
    else:
        raise ValueError('No such model %s' % name)
    return model


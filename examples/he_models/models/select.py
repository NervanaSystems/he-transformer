import tensorflow as tf

import models.cryptodl.cryptodl as cryptodl
import models.simple.simple as simple

tf.app.flags.DEFINE_string('model', 'CryptoDL',
        """One of [CryptoDL].""")

def by_name(name, bool_training=False):
    name = name.lower()
    if name == 'cryptodl':
        model = cryptodl.CryptoDL(bool_training=bool_training)
    elif name == 'simple':
        model = simple.Simple(bool_training=bool_training)
    else:
        raise ValueError('No such model %s' % name)
    return model


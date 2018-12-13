import tensorflow as tf

import models.cryptodl.cryptodl as cryptodl
import models.simple.simple as simple

tf.app.flags.DEFINE_string('model', 'CryptoDL',
        """One of [CryptoDL].""")

def by_name(name, training=False):
    name = name.lower()
    if name == 'cryptodl':
        model = cryptodl.CryptoDL(training=training)
    elif name == 'simple':
        model = simple.Simple(training=training)
    else:
        raise ValueError('No such model %s' % name)
    return model


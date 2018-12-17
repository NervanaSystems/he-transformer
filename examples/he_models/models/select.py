import tensorflow as tf

import models.cryptodl.cryptodl as cryptodl
import models.simple.simple as simple
import models.simple2 as simple2

tf.app.flags.DEFINE_string('model', 'CryptoDL',
        """One of [CryptoDL].""")

def by_name(name, training=False):
    name = name.lower()
    if name == 'cryptodl':
        model = cryptodl.CryptoDL(training=training)
    elif name == 'simple':
        model = simple.Simple(training=training)
    elif name == 'simple2':
        model = simple2.Simple2(training=training)
    else:
        raise ValueError('No such model %s' % name)
    return model


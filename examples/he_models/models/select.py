import tensorflow as tf

import models.cryptodl.cryptodl as cryptodl
import models.simple.simple as simple
import models.simple2 as simple2
import models.simple3 as simple3
import models.simple4 as simple4
import models.simple5 as simple5
import models.test_avg_pool as test_avg_pool

tf.app.flags.DEFINE_string('model', 'CryptoDL', """One of [CryptoDL].""")


def by_name(name, training=False):
    name = name.lower()
    if name == 'cryptodl':
        model = cryptodl.CryptoDL(training=training)
    elif name == 'simple':
        model = simple.Simple(training=training)
    elif name == 'simple2':
        model = simple2.Simple2(training=training)
    elif name == 'simple3':
        model = simple3.Simple3(training=training)
    elif name == 'test_avg_pool':
        model = test_avg_pool.TestAvgPool(training=training)
    elif name == 'simple4':
        model = simple4.Simple4(training=training)
    elif name == 'simple5':
        model = simple5.Simple5(training=training)
    else:
        raise ValueError('No such model %s' % name)
    return model

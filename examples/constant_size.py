# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

import ngraph_bridge
import numpy as np
import tensorflow as tf

print(ngraph_bridge.list_backends())

ngraph_bridge.set_backend('HE_SEAL_CKKS')
print(ngraph_bridge.get_currently_set_backend_name())

batch_size = 500

cifar10_shape = (batch_size, 32, 32, 3)
cifar10_dummy = np.random.random((batch_size, 32, 32, 3))

a = tf.constant(np.ones(cifar10_shape), dtype=np.float32)
b = tf.placeholder(tf.float32, shape=(cifar10_shape))
f = (a + b) * b

with tf.Session() as sess:
    f_val = sess.run(f, feed_dict={b: cifar10_dummy})
    print("Result: ", f_val)

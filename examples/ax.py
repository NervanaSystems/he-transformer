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

a = tf.constant(np.array(range(16)).reshape([2, 2, 2, 2]), dtype=np.float32)
b = tf.placeholder(tf.float32, shape=(2, 2, 2, 2))
k = 2
f = tf.nn.max_pool(
    a + b, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

with tf.Session() as sess:
    f_val = sess.run(f, feed_dict={b: np.ones((2, 2, 2, 2))})
    print("Result: ", f_val)

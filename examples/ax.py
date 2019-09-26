# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
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
from tensorflow.core.protobuf import rewriter_config_pb2

a = tf.constant(np.array([[1, 2, 3, 4]]), dtype=np.float32)
b = tf.compat.v1.placeholder(tf.float32, shape=(1, 4))
c = tf.compat.v1.placeholder(tf.float32, shape=(1))
f = (a + b) * a + c

rewriter_options = rewriter_config_pb2.RewriterConfig()
rewriter_options.meta_optimizer_iterations = (
    rewriter_config_pb2.RewriterConfig.ONE)
rewriter_options.min_graph_nodes = -1
ngraph_optimizer = rewriter_options.custom_optimizers.add()
ngraph_optimizer.name = "ngraph-optimizer"
ngraph_optimizer.parameter_map["ngraph_backend"].s = b'HE_SEAL'
ngraph_optimizer.parameter_map["device_id"].s = b''
ngraph_optimizer.parameter_map[str(b)].s = b'encrypt'

config = tf.compat.v1.ConfigProto()
config.MergeFrom(
    tf.compat.v1.ConfigProto(
        graph_options=tf.compat.v1.GraphOptions(
            rewrite_options=rewriter_options)))

print('config', config)

with tf.compat.v1.Session(config=config) as sess:
    f_val = sess.run(f, feed_dict={b: np.ones((1, 4)), c: np.ones((1))})
    print("Result: ", f_val)

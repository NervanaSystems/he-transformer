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
import time
import sys
import argparse


def gemm_trial(n, p_ones=0, fname="./results.txt"):
    # Pick values not in {-1, 0, 1} for fair comparison
    ra = np.float32(
        np.random.choice([-5, -4, -3, -2, 2, 3, 4, 5], size=(n, n)))
    rb = np.float32(
        np.random.choice([-5, -4, -3, -2, 2, 3, 4, 5], size=(n, n)))

    # randomly set entries to 1
    n_ones = int(p_ones * n * n)
    one_indices = np.random.choice(n * n, n_ones, replace=False)
    for i in one_indices:
        row = i // n
        col = i % n
        ra[row, col] = 1.

    print('ra', ra)
    print('rb', rb)

    a = tf.constant(ra, dtype=np.float32)
    b = tf.placeholder(tf.float32, shape=(n, n))
    c = tf.placeholder(tf.float32, shape=())

    f = tf.matmul(a, b) + c

    with tf.Session() as sess:
        t0 = time.time()
        f_val = sess.run(f, feed_dict={b: rb, c: 3.})
        t1 = time.time() - t0
        print("Result: ", f_val)
        print("Time: ", t1)
        fp = open(fname, 'a')
        fp.write('%d, %f, %f\n' % (n, p_ones, t1))
        fp.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Rum GEMM timing experiments for he-transformer.')
    parser.add_argument(
        '--out', help='Name of output CSV file', default='./results.txt')
    args = parser.parse_args()

    fname = args.out

    ns = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    for n in ns:
        for p_ones in [0, 0.5, 0.8]:
            gemm_trial(n=n, p_ones=p_ones, fname=fname)

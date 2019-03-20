# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
from datetime import datetime
import os

import models.data as data
import models.select as select


# Forward pass = bai
# Backward pass = dy.
@tf.custom_gradient
def straight_through_estimator(x):
    def grad(dy):
        return dy

    return tf.sign(x), grad


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_freq', 10,
                            """How often to log results (steps).""")
tf.app.flags.DEFINE_integer('save_freq', 60,
                            """How often to save model to disk (seconds).""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Continue training the previous model""")
tf.app.flags.DEFINE_string('log_dir', './log/cifar10', "   "
                           "Directory where to write event logs."
                           "")
tf.app.flags.DEFINE_boolean('clip_grads', True,
                            """Clip gradients to [-0.25, 0.25] or not""")
tf.app.flags.DEFINE_boolean('moving_averages', False,
                            """Use moving averages for loss""")

MOVING_AVERAGE_DECAY = 0.9999


def get_run_dir(log_dir, model_name):
    model_dir = os.path.join(log_dir, model_name)
    if FLAGS.batch_norm:
        model_dir += '_bn'
    if FLAGS.train_poly_act:
        model_dir += '_train_poly_act'
    if os.path.isdir(model_dir):
        if FLAGS.resume:
            # Reuse the last directory
            run = len(os.listdir(model_dir)) - 1
        else:
            # We will create a new directory for this run
            run = len(os.listdir(model_dir))
    else:
        run = 0
    return os.path.join(model_dir, '%d' % run)


# Return train_op, loss,
def train_ops():
    # Get training parameters
    data_dir = FLAGS.data_dir
    batch_size = FLAGS.batch_size

    # Create global step counter
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Instantiate async producers for images and labels
    images, labels = data.train_inputs(data_dir=data_dir)

    # Instantiate the model
    model = select.by_name(FLAGS.model, FLAGS, training=True)

    # Create a 'virtual' graph node based on images that represents the input
    # node to be used for graph retrieval
    inputs = tf.identity(images, 'XXX')

    # Build a Graph that computes the logits predictions from the
    # inference model
    logits = model.inference(inputs)
    print('Multiplicative depth', model.mult_depth())

    # In the same way, create a 'virtual' node for outputs
    outputs = tf.identity(logits, 'YYY')

    # Calculate loss
    loss = model.loss(logits, labels)

    # Evaluate training accuracy
    accuracy = model.accuracy(logits, labels)

    # Attach a scalar summary only to the total loss
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('batch accuracy', accuracy)
    # Note that for debugging purpose, we could also track other losses
    for l in tf.get_collection('losses'):
        tf.summary.scalar(l.op.name, l)

    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Clip gradients to [-0.25, 0.25]
    if FLAGS.clip_grads:
        print("Clipping gradients to [-0.25, 0.25]")
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = []
        for grad, var in gvs:
            if grad is None:
                continue
            capped_gvs.append((tf.clip_by_value(grad, -0.25, 0.25), var))
        sgd_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    else:
        print("Not clipping gradients")
        sgd_op = optimizer.minimize(loss, global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Create a meta-graph that includes sgd and variables moving average
    with tf.control_dependencies([sgd_op] + update_ops):
        train_op = tf.no_op(name='train')

    return (train_op, loss, accuracy)


def train_loop():
    train_op, loss, accuracy = train_ops()

    run_dir = get_run_dir(FLAGS.log_dir, FLAGS.model)
    checkpoint_dir = os.path.join(run_dir, 'train')

    # This class implements the callbacks for the logger
    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""

        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs([loss,
                                            accuracy])  # Asks for values.

        def after_run(self, run_context, run_values):
            if self._step % FLAGS.log_freq == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value = run_values.results[0]
                accuracy_value = run_values.results[1]
                examples_per_sec = FLAGS.log_freq * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_freq)

                format_str = '%s: step %d, loss = %.2f , accuracy = %.2f '
                format_str += '(%.1f examples/sec; %.3f sec/batch)'
                print(format_str %
                      (datetime.now(), self._step, loss_value, accuracy_value,
                       examples_per_sec, sec_per_batch))

    # Start the training loop using a monitored session (automatically takes
    # care of thread sync)
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_secs=FLAGS.save_freq,
            hooks=[
                tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                tf.train.NanTensorHook(loss),
                _LoggerHook()
            ]) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)


def main(argv=None):
    data.maybe_download_and_extract(FLAGS.data_dir)
    train_loop()


if __name__ == '__main__':
    tf.app.run()

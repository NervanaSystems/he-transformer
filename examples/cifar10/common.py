import tensorflow as tf
import numpy as np

def get_variable(name, shape, dtype=None, initializer=None, restore_saved=False):
  if not restore_saved:
    return tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer)
  else:
    restore_filename = name_to_filename(name)
    print('restoring variable: ', restore_filename)
    return tf.constant(np.loadtxt(restore_filename, dtype=np.float32).reshape(shape))



def fire_layer(inputs, s1x1, e1x1, e3x3, name, decay=False, restore_saved=False):
  with tf.variable_scope(name) as scope:
      # Squeeze sub-layer
      squeezed_inputs = conv_layer(inputs,
                                        size=1,
                                        filters=s1x1,
                                        stride=1,
                                        decay=decay,
                                        name='s1x1',
                                        restore_saved=restore_saved)

      # Expand 1x1 sub-layer
      e1x1_outputs = conv_layer(squeezed_inputs,
                                      size=1,
                                      filters=e1x1,
                                      stride=1,
                                      decay=decay,
                                      name='e1x1',
                                      restore_saved=restore_saved)

      # Expand 3x3 sub-layer
      e3x3_outputs = conv_layer(squeezed_inputs,
                                      size=3,
                                      filters=e3x3,
                                      stride=1,
                                      decay=decay,
                                      name='e3x3',
                                      restore_saved=restore_saved)

  # Concatenate outputs along the last dimension (channel)
  return tf.concat([e1x1_outputs, e3x3_outputs], 3)

def name_to_filename(name):
  name = name.replace('/','_')
  name = name.replace(':0', '') + '.txt'

  return 'weights/' + name


def conv_layer(inputs, size, filters, stride, decay, name, bn=False, restore_saved=False):
  channels = inputs.get_shape()[3]
  shape = [size, size, channels, filters]
  with tf.variable_scope(name + '/conv') as scope:
    if not restore_saved:
      weights = get_variable('weights', shape=shape)
      biases = get_variable('biases',
                            shape=[filters],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
      conv = tf.nn.conv2d(inputs,
                        weights,
                        strides=[1,stride,stride,1],
                        padding='SAME')
      pre_act = tf.nn.bias_add(conv, biases)

      a = get_variable('a', shape=[1], initializer=tf.constant_initializer(0.0))
      b = get_variable('b', shape=[1])

      outputs = pre_act * pre_act * a + b * pre_act
      # outputs = tf.nn.relu(pre_activation, name=scope.name)

    else: # Restore saved
      print('scope', scope.name)

      weights_name =scope.name + '/weights'
      biases_name = scope.name + '/biases'
      a_name = scope.name + '/a'
      b_name = scope.name + '/b'

      weights = get_variable(weights_name, shape=shape, restore_saved=True)

      conv = tf.nn.conv2d(inputs,
                        weights,
                        strides=[1,stride,stride,1],
                        padding='SAME')

      biases = get_variable(biases_name,
                              shape=[filters],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.0), restore_saved=True)

      pre_act = tf.nn.bias_add(conv, biases)

      a = get_variable(a_name, shape=[1], initializer=tf.constant_initializer(0.0), restore_saved=True)
      b = get_variable(b_name, shape=[1], restore_saved=True)

      pre_act = tf.nn.bias_add(conv, biases)
      outputs = pre_act * pre_act * a + b * pre_act

  return outputs

def pool_layer(inputs, size, stride, name):
    with tf.variable_scope(name) as scope:
        outputs = tf.nn.avg_pool(inputs,
                                  ksize=[1,size,size,1],
                                  strides=[1,stride,stride,1],
                                  padding='SAME',
                                  name=name)

    return outputs

def avg_layer(inputs, name):
  w = inputs.get_shape().as_list()[1]
  h = inputs.get_shape().as_list()[2]
  c = inputs.get_shape().as_list()[3]
  with tf.variable_scope(name) as scope:
      # Use current spatial dimensions as Kernel size to produce a scalar
      avg = tf.nn.avg_pool(inputs,
                            ksize=[1,w,h,1],
                            strides=[1,1,1,1],
                            padding='VALID',
                            name=scope.name)
  # Reshape output to remove spatial dimensions reduced to one
  return tf.reshape(avg, shape=[-1,c])
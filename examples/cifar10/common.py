import tensorflow as tf

def fire_layer(inputs, s1x1, e1x1, e3x3, name, decay=False):
  with tf.variable_scope(name) as scope:
      # Squeeze sub-layer
      squeezed_inputs = conv_layer(inputs,
                                        size=1,
                                        filters=s1x1,
                                        stride=1,
                                        decay=decay,
                                        name='s1x1')

      # Expand 1x1 sub-layer
      e1x1_outputs = conv_layer(squeezed_inputs,
                                      size=1,
                                      filters=e1x1,
                                      stride=1,
                                      decay=decay,
                                      name='e1x1')

      # Expand 3x3 sub-layer
      e3x3_outputs = conv_layer(squeezed_inputs,
                                      size=3,
                                      filters=e3x3,
                                      stride=1,
                                      decay=decay,
                                      name='e3x3')

  # Concatenate outputs along the last dimension (channel)
  return tf.concat([e1x1_outputs, e3x3_outputs], 3)

def conv_layer(inputs, size, filters, stride, decay, name, bn=False):
  channels = inputs.get_shape()[3]
  shape = [size, size, channels, filters]
  with tf.variable_scope(name + '/conv') as scope:
      weights = tf.get_variable('weights', shape=shape)

      biases = tf.get_variable('biases',
                              shape=[filters],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.0))
      conv = tf.nn.conv2d(inputs,
                          weights,
                          strides=[1,stride,stride,1],
                          padding='SAME')
      pre_activation = tf.nn.bias_add(conv, biases)

      a = tf.get_variable('a', shape=[1], initializer=tf.constant_initializer(0.0))
      b = tf.get_variable('b', shape=[1])

      outputs = a * pre_activation**2 + b * pre_activation

      outputs = tf.nn.relu(pre_activation, name=scope.name)

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
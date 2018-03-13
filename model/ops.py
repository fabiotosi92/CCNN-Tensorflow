import tensorflow as tf


def conv2d(x, kernel_shape, strides=1, relu=True, padding='SAME'):
    W = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.get_variable("biases", kernel_shape[3], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    with tf.name_scope("conv"):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        tf.summary.histogram("W", W)
        tf.summary.histogram("b", b)
        if kernel_shape[2] == 3:
            x_min = tf.reduce_min(W)
            x_max = tf.reduce_max(W)
            kernel_0_to_1 = (W - x_min) / (x_max - x_min)
            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
            tf.summary.image('filters', kernel_transposed, max_outputs=3)
        if relu:
            x =  tf.nn.relu(x)
    return x
import tensorflow as tf
import numpy as np
from tflearn.layers.conv import global_avg_pool


def Fully_connected(x, units, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=True, units=units)


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

class ResNet(object):
    def __init__(self, input_shape, class_num=164, training=True):
        self.height, self.width = input_shape
        self.class_num = class_num
        self.training = training
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 3], name='input_holder')
        self.ground_truth = tf.placeholder(tf.float32, shape=[None, class_num])

    def Squeeze_excitation_layer(self, input_x, layer_name, ratio=4):
        with tf.name_scope(layer_name):
            out_dim = int(np.shape(input_x)[-1])
            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
            excitation = tf.nn.relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
            excitation = tf.nn.sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation

            return scale


    def residual_block(self, inputs, output_channel, inner_depth, stride, index):
        input_channel = inputs.get_shape().as_list()[-1]
        with tf.variable_scope('bottleneck_{}'.format(index)):
            pre_act = tf.layers.batch_normalization(inputs, momentum=0.9, name='preact')
            pre_act = tf.nn.relu(pre_act)

            if input_channel == output_channel:
                shortcut = inputs
                if stride == 2:
                    shortcut = tf.layers.max_pooling2d(shortcut,
                                            pool_size=(2, 2),
                                            strides=(stride, stride),
                                            name='shortcut')
            else:
                shortcut = tf.layers.conv2d(pre_act,
                                            filters=output_channel,
                                            kernel_size=[1, 1],
                                            strides=[stride, stride],
                                            padding='SAME',
                                            name='shortcut')
            # tf.layers.batch_normalization()
            residual = tf.layers.conv2d(pre_act,
                                        filters=inner_depth,
                                        kernel_size=[1, 1],
                                        strides=[1, 1],
                                        use_bias=False,
                                        padding='SAME',
                                        name='conv1')
            residual = tf.layers.batch_normalization(residual, momentum=0.9, name='conv1')
            if stride == 1:
                residual = tf.layers.conv2d(residual,
                                            filters=inner_depth,
                                            kernel_size=[3, 3],
                                            strides=[stride, stride],
                                            padding='SAME',
                                            use_bias=False,
                                            name='conv2')
            else:
                pad_total = 3 - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                inputs = tf.pad(inputs,
                                [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
                residual = tf.layers.conv2d(inputs,
                                            filters=inner_depth,
                                            kernel_size=[3, 3],
                                            strides=[stride, stride],
                                            padding='VALID',
                                            use_bias=False,
                                            name='conv2')
            residual = tf.layers.batch_normalization(residual, momentum=0.9, name='conv2')

            residual = tf.layers.conv2d(residual,
                                        filters=output_channel,
                                        kernel_size=[1, 1],
                                        strides=[1, 1],
                                        padding='SAME',
                                        name='conv3')
            print(shortcut, residual)
            output = shortcut + residual
        return output

    def build(self):
        self.conv1 = tf.layers.conv2d(self.inputs, filters=64, kernel_size=[7, 7], strides=[2, 2], padding='SAME', name='conv1')
        self.pool1 = tf.layers.max_pooling2d(self.conv1, pool_size=[3, 3], strides=[2, 2], padding='SAME', name='pool1')
        net = self.pool1
        with tf.variable_scope('residual_1'):
            for i in range(2):
                net = self.residual_block(net, 256, 64, 1, i)
            net = self.residual_block(net, 256, 64, 2, 2)

        with tf.variable_scope('residual_2'):
            for i in range(3):
                net = self.residual_block(net, 512, 128, 1, i)
            net = self.residual_block(net, 512, 128, 2, 3)

        with tf.variable_scope('residual_3'):
            for i in range(5):
                net = self.residual_block(net, 1024, 256, 1, i)
            net = self.residual_block(net, 1024, 256, 2, 5)

        with tf.variable_scope('residual_4'):
            for i in range(3):
                net = self.residual_block(net, 2048, 512, 1, i)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net = tf.reduce_mean(net, [1, 2], name='pool_5', keep_dims=True)
        self.out_put = tf.reshape(net,[-1, 2048])
        net = tf.layers.dropout(net, rate=0.2, training=True)
        net = tf.layers.conv2d(net, self.class_num, padding='SAME', kernel_size=[1, 1], name='logits')
        self.result = tf.squeeze(net, axis=[1, 2])
        return self.result, self.out_put

    def loss(self):
        print(self.ground_truth)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.result, labels=self.ground_truth), name='loss')



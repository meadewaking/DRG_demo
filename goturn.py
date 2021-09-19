import tensorflow as tf
import numpy as np


class Goturn(object):

    def __init__(self):
        self.fc_four, self.x, self.target, self.target_net, self.feature, self.gt = self.main_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        ckpt_path = 'model/'
        self.checkpoint = tf.train.get_checkpoint_state(ckpt_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.checkpoint.model_checkpoint_path)
        print('init')

    def conv2d(self, x, weights, strides, pad=0, trainable=False, name=None, iter=1, bias_init=0.0):
        with tf.variable_scope(name):
            if pad > 0:
                x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

            weight = tf.Variable(tf.truncated_normal(weights, dtype=tf.float32, stddev=1e-2), trainable=trainable,
                                 name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[weights[3]], dtype=tf.float32), trainable=trainable,
                                 name='biases')

            if iter == 1:
                net = tf.nn.bias_add(tf.nn.conv2d(x, weight, strides, padding='VALID'), biases)
            elif iter == 2:
                w1, w2 = tf.split(weight, 2, axis=3)
                x1, x2 = tf.split(x, 2, axis=3)
                conv_one = tf.nn.conv2d(x1, w1, strides, padding='VALID')
                conv_two = tf.nn.conv2d(x2, w2, strides, padding='VALID')
                net = tf.nn.bias_add(tf.concat([conv_one, conv_two], axis=3), biases)

            net = tf.nn.relu(net)

            return net

    def caffenet(self, x, name):

        net = self.conv2d(x, [11, 11, 3, 96], [1, 4, 4, 1], name=name + '_conv_1')
        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name=name + '_pool1')
        net = tf.nn.local_response_normalization(net, depth_radius=2, alpha=1e-4, beta=0.75, name=name + '_lrn1')
        net = self.conv2d(net, [5, 5, 48, 256], [1, 1, 1, 1], pad=2, iter=2, bias_init=1.0, name=name + '_conv_2')
        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name=name + '_pool2')
        net = tf.nn.local_response_normalization(net, depth_radius=2, alpha=1e-4, beta=0.75, name=name + '_lrn2')
        net = self.conv2d(net, [3, 3, 256, 384], [1, 1, 1, 1], pad=1, name=name + '_conv_3')
        net = self.conv2d(net, [3, 3, 192, 384], [1, 1, 1, 1], pad=1, iter=2, name=name + '_conv_4')
        net = self.conv2d(net, [3, 3, 192, 256], [1, 1, 1, 1], pad=1, iter=2, bias_init=1.0, name=name + '_conv_5')
        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name=name + '_pool5')

        return net

    def fc_layer(self, x, c, name, af=True):
        with tf.variable_scope(name):
            shape = int(np.prod(x.get_shape()[1:]))
            w = tf.Variable(tf.truncated_normal([shape, c], dtype=tf.float32, stddev=0.001), name='weights')
            b = tf.Variable(tf.constant(0.1, shape=[c], dtype=tf.float32), name='biases')
            x = tf.reshape(x, [-1, shape])
            out = tf.nn.bias_add(
                tf.matmul(x, w), b
            )
            if af:
                return tf.nn.relu(out)
            else:
                return out

    def main_net(self):

        x = tf.placeholder(tf.float32, [None, 227, 227, 3])
        target = tf.placeholder(tf.float32, [None, 227, 227, 3])
        feature = tf.placeholder(tf.float32, [None, 36, 1, 256])
        gt = tf.placeholder(tf.float32, [None, 1])

        target_net = self.caffenet(target, 'target')
        img_net = self.caffenet(x, 'image')
        net = tf.concat([target_net, img_net], axis=3)
        net = tf.transpose(net, (0, 3, 1, 2))
        fc_one = self.fc_layer(net, 4096, 'fc1')
        fc_two = self.fc_layer(fc_one, 4096, 'fc2')
        fc_three = self.fc_layer(fc_two, 4096, 'fc3')
        fc_four = self.fc_layer(fc_three, 1, 'fc4', False)

        return fc_four, x, target, target_net, feature, gt

    def get_output(self, state):
        out = self.sess.run(self.fc_four, feed_dict={
            self.x: state,
            self.target: state
        })
        return out

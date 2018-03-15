# train_occ_new.py

import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

import utils
from base import Base


class Train_occ(Base):
    def __init__(self, base_dir='../../../Research/MPI-Sintel/'):
        Base.__init__(self)
        self.data_img = None
        self.data_occ = None
        self.base_dir = base_dir
        self.batch_size = 15

    def train_main(self):
        print('train started')

        if self.data_img is None or self.data_occ is None:
            self.load_data()
        if self.data_img is None or self.data_img.shape[0] == 0:
            print('cannot load img data')
            return

        sess = tf.Session()

        n_ch = 1
        x = tf.placeholder(tf.float32, shape=[None, 65536 * 2 * n_ch])
        y_ = tf.placeholder(tf.float32, shape=[None, 65536])

        x_img = tf.reshape(x, [self.batch_size, 256, 256, 2 * n_ch])
        y_img = tf.reshape(y_, [self.batch_size, 256, 256, 1])

        x_img_0 = tf.reshape(x_img[:, :, :, 0], [self.batch_size, 256, 256, 1])
        x_img_1 = tf.reshape(x_img[:, :, :, 1 * n_ch], [self.batch_size, 256, 256, 1])
        x_img_diff = x_img_0 - x_img_1

        print('-- convolutions ---')

        # net_0 = self.conv(x_img_0, 1, 16, strides=[1, 2, 2, 1])
        # print('h_conv1_0: ', net_0)
        # net_1 = self.conv(x_img_1, 1, 16, strides=[1, 2, 2, 1])
        # print('h_conv1_1: ', net_1)

        # net_0 = self.conv(net_0, 16, 32, strides=[1, 2, 2, 1])
        # print('h_conv2_0: ', net_0)
        # net_1 = self.conv(net_1, 16, 32, strides=[1, 2, 2, 1])
        # print('h_conv2_1: ', net_1)

        # h_conv2 = tf.concat(3, [net_0, net_1])
        # print('h_conv2: ', h_conv2)

        # CL1 (256 -> 128)
        h_conv0 = self.conv(x_img, 2, 16, strides=[1, 1, 1, 1], ksize=3)
        h_conv0 = self.conv(h_conv0, 16, 16, strides=[1, 1, 1, 1], ksize=1)
        h_conv1 = self.conv(h_conv0, 16, 32, strides=[1, 2, 2, 1], ksize=3)
        print('h_conv1: ', h_conv1)

        # CL2 (128 -> 64)
        h_conv2_h = self.conv(h_conv1, 32, 32, strides=[1, 1, 1, 1], ksize=1)
        h_conv2_h = self.conv(h_conv2_h, 32, 32, strides=[1, 1, 1, 1], ksize=1)
        h_conv2 = self.conv(h_conv2_h, 32, 64, strides=[1, 2, 2, 1], ksize=3)
        print('h_conv2: ', h_conv2)

        # CL3 (64 -> 32)
        h_conv3_h = self.conv(h_conv2, 64, 64, strides=[1, 1, 1, 1], ksize=1)
        h_conv3_h = self.conv(h_conv3_h, 64, 64, strides=[1, 1, 1, 1], ksize=1)
        h_conv3 = self.conv(h_conv3_h, 64, 128, strides=[1, 2, 2, 1], ksize=3)
        print('h_conv3: ', h_conv3)

        # CL4 (32 -> 16)
        h_conv4_h = self.conv(h_conv3, 128, 128, strides=[1, 1, 1, 1], ksize=1)
        h_conv4_h = self.conv(h_conv4_h, 128, 128, strides=[1, 1, 1, 1], ksize=1)
        h_conv4 = self.conv(h_conv4_h, 128, 256, strides=[1, 2, 2, 1], ksize=3)
        print('h_conv4: ', h_conv4)

        print('-- densely connected layer ---')
        W_dl = self.weight_variable([16 * 16 * 256, 1024])
        b_dl = self.bias_variable([1024])

        h_flat = tf.reshape(h_conv4, [self.batch_size, 16 * 16 * 256])
        h_dl = tf.nn.relu(tf.matmul(h_flat, W_dl) + b_dl)
        print('h_dl: ', h_dl)

        # h_dl = tf.nn.dropout(h_dl, 0.5)

        # W_cat = self.weight_variable([1024, 2])
        # b_cat = self.bias_variable([2])
        # h_cat = tf.matmul(h_dl, W_cat) + b_cat
        # print('h_cat: ', h_cat)

        # W_icat = self.weight_variable([2, 1024])
        # b_icat = self.bias_variable([1024])
        # h_icat = tf.matmul(h_cat, W_icat) + b_icat
        # print('h_icat: ', h_icat)

        W_idl = self.weight_variable([1024, 16 * 16 * 256])
        b_idl = self.bias_variable([16 * 16 * 256])
        h_idl = tf.nn.relu(tf.matmul(h_dl, W_idl) + b_idl)
        h_idl = tf.reshape(h_idl, [self.batch_size, 16, 16, 256])
        print('h_idl: ', h_idl)

        print('-- deconvolutions ---')

        h_dconv4_c = tf.concat(3, [h_idl, h_conv4])
        print('h_dconv4_c: ', h_dconv4_c)
        h_dconv4_h = self.deconv(h_dconv4_c, 512, 256, [self.batch_size, 16, 16, 256],
                                 strides=[1, 1, 1, 1], ksize=3)
        h_dconv4_h = self.deconv(h_dconv4_h, 256, 256, [self.batch_size, 16, 16, 256],
                                 strides=[1, 1, 1, 1], ksize=1)
        h_dconv4 = self.deconv(h_dconv4_h, 256, 128, [self.batch_size, 32, 32, 128],
                               strides=[1, 2, 2, 1], ksize=3)
        print('h_dconv4:', h_dconv4)

        h_dconv3_c = tf.concat(3, [h_dconv4, h_conv3])
        print('h_dconv3_c: ', h_dconv3_c)
        h_dconv3_h = self.deconv(h_dconv3_c, 256, 128, [self.batch_size, 32, 32, 128],
                                 strides=[1, 1, 1, 1], ksize=3)
        h_dconv3_h = self.deconv(h_dconv3_h, 128, 128, [self.batch_size, 32, 32, 128],
                                 strides=[1, 1, 1, 1], ksize=1)
        h_dconv3 = self.deconv(h_dconv3_h, 128, 64, [self.batch_size, 64, 64, 64],
                               strides=[1, 2, 2, 1], ksize=3)
        print('h_dconv3:', h_dconv3)

        h_dconv2_c = tf.concat(3, [h_dconv3, h_conv2])
        print('h_dconv2_c: ', h_dconv2_c)
        h_dconv2_h = self.deconv(h_dconv2_c, 128, 64, [self.batch_size, 64, 64, 64],
                                 strides=[1, 1, 1, 1], ksize=3)
        h_dconv2_h = self.deconv(h_dconv2_h, 64, 64, [self.batch_size, 64, 64, 64],
                                 strides=[1, 1, 1, 1], ksize=1)
        h_dconv2 = self.deconv(h_dconv2_h, 64, 32, [self.batch_size, 128, 128, 32],
                               strides=[1, 2, 2, 1], ksize=3)
        print('h_dconv2: ', h_dconv2)

        h_dconv1_c = tf.concat(3, [h_dconv2, h_conv1])
        print('h_dconv1_c: ', h_dconv1_c)
        h_dconv1_h = self.deconv(h_dconv1_c, 64, 32, [self.batch_size, 128, 128, 32],
                                 strides=[1, 1, 1, 1], ksize=3)
        h_dconv1_h = self.deconv(h_dconv1_h, 32, 32, [self.batch_size, 128, 128, 32],
                                 strides=[1, 1, 1, 1], ksize=1)
        h_dconv1 = self.deconv(h_dconv1_h, 32, 16, [self.batch_size, 256, 256, 16],
                               strides=[1, 2, 2, 1], ksize=3)
        print('h_dconv1: ', h_dconv1)

        # h_dconv0_c = tf.concat(3, [h_dconv1, h_conv0])
        # h_dconv0 = self.deconv(h_dconv0_c, 32, 16, [self.batch_size, 256, 256, 16],
        #                        strides=[1, 1, 1, 1], ksize=1)
        h_dconv0_2 = h_dconv1
        h_dconv0_2 = self.deconv(h_dconv0_2, 16, 2, [self.batch_size, 256, 256, 2],
                                 strides=[1, 1, 1, 1], ksize=1)
        h_dconv0 = self.deconv(h_dconv1, 16, 1, [self.batch_size, 256, 256, 1],
                               strides=[1, 1, 1, 1], ksize=3)
        print('h_dconv0: ', h_dconv0)

        rst_img_0 = h_dconv0_2[0, :, :, 0]

        h_dconv0_flat = tf.reshape(h_dconv0, [self.batch_size, 65536])

        # # cost = tf.reduce_mean(tf.square(y_ - h_dconv0_flat))
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h_dconv0_flat, y_))
        # train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)
        # # accuracy = tf.reduce_mean(tf.square(y_ - h_dconv0_flat))
        # accuracy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h_dconv0_flat, y_))

        sequence_lengths = np.full(1, 1, dtype=np.int32)
        sequence_lengths_t = tf.constant(sequence_lengths)

        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(h_dconv0_flat, y_, sequence_lengths_t)
        loss = tf.reduce_mean(-log_likelihood)
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        accuracy = tf.reduce_mean(-log_likelihood)

        sess.run(tf.initialize_all_variables())

        fig = self.init_figure(figsize=(10, 5))
        rn, cn = 1, 3

        max_cycles = int(18000 / float(self.batch_size))
        for i in range(max_cycles):
            batch = []
            batch.append(self.data_img[i * self.batch_size:i * self.batch_size + self.batch_size])
            batch.append(self.data_occ[i * self.batch_size:i * self.batch_size + self.batch_size])
            print('ep %d: %d' % (i, len(batch[0])))
            # train_step.run(feed_dict={x: batch[0], y_: batch[1]})

            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
            if i % 2 == 0:
                # train_accuracy = accuracy.eval(feed_dict={
                #     x: batch[0], y_: batch[1]})
                train_accuracy, img, rst, tch, gt = sess.run([accuracy, x_img_diff, h_dconv0,
                                                             rst_img_0, y_img],
                                                             feed_dict={x: batch[0], y_: batch[1]})
                print("step %d, training accuracy: %g" % (i, train_accuracy))

                img = img[0, :, :, 0]
                rst = rst[0, :, :, 0]
                gt = gt[0, :, :, 0]
                # img[img < 30] = 0
                # img[img >= 30] = 1
                # gt[gt < 200] = 0
                # gt[gt >= 200] = 1

                counter = 0
                # for ridx in range(256):
                #     for cidx in range(256):
                #         if img[ridx][cidx] == gt[ridx][cidx]:
                #             counter += 1
                print("step %d, dist: %g, training accuracy: %g"
                      % (i, train_accuracy, counter / 65536.))

                fig.clear()
                utils.imshow_in_subplot(rn, cn, 1, tch)
                utils.imshow_in_subplot(rn, cn, 2, rst)
                utils.imshow_in_subplot(rn, cn, 3, gt)
                utils.save_fig_in_dir(fig, dirname='rst', filename='batch_%04d.png' % i)

        print('train finished')

    def load_data(self):
        self.data_img, self.data_occ = utils.load_data(self.base_dir)
        print(self.data_img.shape, self.data_occ.shape)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv(self, x, ch_in, ch_out, strides=[1, 2, 2, 1], ksize=3):
        W = self.weight_variable([ksize, ksize, ch_in, ch_out])
        b = self.bias_variable([ch_out])
        h_conv = tf.nn.relu(self.conv2d(x, W, strides=strides) + b)
        return h_conv

    def conv2d(self, x, W, strides=[1, 2, 2, 1]):
        return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

    def deconv(self, x, ch_in, ch_out, out_shape, strides=[1, 2, 2, 1], ksize=3):
        W = self.weight_variable([ksize, ksize, ch_out, ch_in])
        b = self.bias_variable([ch_out])
        h_dconv = tf.nn.relu(self.deconv2d(x, W, out_shape, strides=strides) + b)
        return h_dconv

    def deconv2d(self, x, W, out_shape, strides=[1, 2, 2, 1]):
        return tf.nn.conv2d_transpose(x, W, output_shape=out_shape,
                                      strides=strides, padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    train_occ = Train_occ('../../MPI-Sintel/')
    train_occ.train_main()

    # img_idxs = [600, 601, 602, 603, 604, 605]
    # for img_idx in img_idxs:
    #     print(train_occ.data_img[img_idx, 0].shape)
    #     img = np.reshape(train_occ.data_img[img_idx], (200, 200, 2))
    #     img_l = img[:, :, 0]
    #     img_r = img[:, :, 1]
    #     occ = np.reshape(train_occ.data_occ[img_idx], (200, 200))
    #     train_occ.init_figure(figsize=(15, 5))
    #     rn, cn = 1, 3
    #     utils.imshow_in_subplot(rn, cn, 1, img_l)
    #     utils.imshow_in_subplot(rn, cn, 2, img_r)
    #     utils.imshow_in_subplot(rn, cn, 3, occ)

    # utils.show_plots()

# End of script

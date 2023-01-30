#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorflow.contrib.layers as ly
import os
import scipy.io as sio
import time
import h5py
import tensorflow as tf
import tensorlayer as tl
from config import *
from tensorlayer.layers import *
from tensorflow.python.framework import graph_util

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            # cv2.boxFilter(data[i, :, :], -1, (5, 5)) 得到低频信息
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (10, 10))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (10, 10))
    return rs


def get_batch(gt, rgb_hp, ms):
    shapes = gt.shape
    batch_index = np.random.randint(0, shapes[0], size=batch_size)

    gt_batch = gt[batch_index, :, :, :]
    rgb_hp_batch = rgb_hp[batch_index, :, :, :]
    ms_batch = ms[batch_index, :, :, :]

    return gt_batch, rgb_hp_batch, ms_batch


def get_all_data(datastr):
    train_data_name = datastr  # training data
    train_data = h5py.File(train_data_name, mode='r')  # 读取mat文件

    gt = train_data['gt'][...]  ## ground truth N*H*W*C
    rgb = train_data['rgb'][...]  #### Pan image N*H*W
    ms = train_data['ms'][...]  ### low resolution MS image
    lms = train_data['lms'][...]  #### MS image interpolation to Pan scale

    gt = np.array(gt, dtype=np.float32) / (2 ** 16 - 1)
    rgb = np.array(rgb, dtype=np.float32) / (2 ** 8 - 1)
    ms = np.array(ms, dtype=np.float32) / (2 ** 16 - 1)
    lms = np.array(lms, dtype=np.float32) / (2 ** 16 - 1)

    rgb_hp = rgb


    return gt, rgb_hp, ms, lms


def HyNetSingleLevel(net_image, net_feature, rgbDownsample, reuse=False):
    with tf.variable_scope("Model_level", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        shapes = net_feature.outputs.shape

        net_feature = DeConv2dLayer(net_feature, shape=[6, 6, 64, 64], strides=[1, 2, 2, 1],
                                    output_shape=(shapes[0], shapes[1] * 2, shapes[2] * 2, 64),
                                    name='deconv_feature', W_init=tf.contrib.layers.xavier_initializer())

        concat_feature = ConcatLayer([net_feature, rgbDownsample], 3, name='concat_layer')

        net_feature = Conv2dLayer(concat_feature, shape=[3, 3, 67, 67], strides=[1, 1, 1, 1],
                                  W_init=tf.contrib.layers.xavier_initializer(), name='conv2D67')
        net_feature = Conv2dLayer(net_feature, shape=[3, 3, 67, 64], strides=[1, 1, 1, 1],
                                  W_init=tf.contrib.layers.xavier_initializer(), name='conv2Dto64')

        for d in range(res_number):

            net_tmp=InputLayer(tl.activation.leaky_relu(net_feature.outputs),name='actinput'+str(d))
            net_tmp = Conv2dLayer(net_tmp, shape=[3, 3, 64, 64], strides=[1, 1, 1, 1],
                                  W_init=tf.contrib.layers.xavier_initializer(), name='conv2D1' + str(d))

            net_feature = ElementwiseLayer([net_tmp, net_feature], combine_fn=tf.add, name='add_temp' + str(d))


        # 残差
        gradient_level = Conv2dLayer(net_feature, shape=[3, 3, 64, 31], strides=[1, 1, 1, 1],
                                     W_init=tf.contrib.layers.xavier_initializer(), name='conv2Dget_gradient')

        imageshape = net_image.outputs.shape

        net_image = Conv2dLayer(net_image,shape=[3,3,31,31*4],strides=[1,1,1,1],
                        name='upconv_image', W_init=tf.contrib.layers.xavier_initializer())
        net_image = SubpixelConv2d(net_image,scale=2,n_out_channel=31,
                        name='subpixel_image')


        net_image = ElementwiseLayer([gradient_level, net_image], combine_fn=tf.add, name='add_image')

    return net_image, net_feature


def HyTestNet(inputs, rgb_image, is_train=False, reuse=False):
    with tf.variable_scope("HyTestNet", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)

        shapes = tf.shape(rgb_image)

        inputs_level = InputLayer(inputs, name='input_level')

        net_feature = Conv2dLayer(inputs_level, shape=[3, 3, 31, 64], strides=[1, 1, 1, 1],
                                  W_init=tf.contrib.layers.xavier_initializer(), name='init_conv')
        net_feature = Conv2dLayer(net_feature, shape=[3, 3, 64, 64], strides=[1, 1, 1, 1],
                                  W_init=tf.contrib.layers.xavier_initializer(), name='init_conv2')
        # (4 10 10 64)
        rgbSample = InputLayer(rgb_image, name='rgb_level')

        rgbSample1 = Conv2dLayer(rgbSample, shape=[6, 6, 3, 3], strides=[1, 2, 2, 1],
                                 W_init=tf.contrib.layers.xavier_initializer(), name='rgbdown1')
        rgbSample2 = Conv2dLayer(rgbSample1, shape=[6, 6, 3, 3], strides=[1, 2, 2, 1],
                                 W_init=tf.contrib.layers.xavier_initializer(), name='rgbdown2')


        net_image = inputs_level
        # 2X for each level
        net_image1, net_feature1 = HyNetSingleLevel(net_image, net_feature, rgbSample2, reuse=reuse)
        net_image2, net_feature2 = HyNetSingleLevel(net_image1, net_feature1, rgbSample1, reuse=True)
        net_image3, net_feature3 = HyNetSingleLevel(net_image2, net_feature2, rgbSample, reuse=True)

    return net_image3, net_image2, net_image1


if __name__ == '__main__':
    tf.reset_default_graph()
    batch_size = 32
    in_patch_size = 10
    scale = 8
    channel = 31

    res_number = 10
    iterations = 350001
    lr_rate = 0.0001
    # lr_rate=0.0001
    # telement_number=1000
    model_directory1 = './models/train_CSR_conv/'  # directory to save trained model to.
    model_directory = './models/train_CSR_conv'  # directory to save trained model to.


    train_data_name = '/Data/Machine_Learning/Ran Ran/data/rgbnet/train/trainCSR2420.mat'  # training data
    test_data_name = '/Data/Machine_Learning/Ran Ran/data/rgbnet/train/validationCSR2420.mat'  # validation data


    restore = False  # load model or not

    method = 'Adam'  # training method: Adam or SGD
    alpha = [1, 2, 4]  # 0: gt 尺寸result,1/2 gt 尺寸result,1/4 GT 尺寸result loss系数
    ############## loading data
    train_gt, train_rgb_hp, train_ms, train_lms = get_all_data(train_data_name)
    test_gt, test_rgb_hp, test_ms, test_lms = get_all_data(test_data_name)



    ############## placeholder for training
    t_image = tf.placeholder('float32', [batch_size, in_patch_size, in_patch_size, channel],
                             name='t_image')
    rgb_image = tf.placeholder('float32', [batch_size, in_patch_size * scale,
                                           in_patch_size * scale, 3], name='rgb_image')
    t_target_image = tf.placeholder('float32', [batch_size, in_patch_size * scale,
                                                in_patch_size * scale, channel], name='t_target_image')

    ############# placeholder for testing
    test_image = tf.placeholder('float32', [batch_size, in_patch_size, in_patch_size, channel],
                                name='test_image')
    testrgb_image = tf.placeholder('float32', [batch_size, in_patch_size * scale,
                                               in_patch_size * scale, 3], name='testrgb_image')
    test_target_image = tf.placeholder('float32', [batch_size, in_patch_size * scale,
                                                   in_patch_size * scale, channel], name='test_target_image')

    ######## network architecture
    net_image3, net_image2, net_image1, = HyTestNet(t_image, rgb_image, is_train=True, reuse=False)
    # time.sleep(100000)
    # 32 64 64 8
    net_testimage3, net_testimage2, net_testimage1, = HyTestNet(test_image, testrgb_image, is_train=False, reuse=True)
    # test_rs = test_rs + test_lms  # same as: test_rs = tf.add(test_rs,test_lms)
    # 32 64 64 8

    ######## loss function
    # (124, 80, 80, 1)
    t_target_image_down1 = tf.image.resize_images(t_target_image,
                                                  size=[in_patch_size * scale // 2, in_patch_size * scale // 2],
                                                  method=0, align_corners=False)
    # (124, 40, 40, 1)
    t_target_image_down2 = tf.image.resize_images(t_target_image,
                                                  size=[in_patch_size * scale // 4, in_patch_size * scale // 4],
                                                  method=0, align_corners=False)


    loss1 = tf.reduce_mean(tf.abs(net_image3.outputs - t_target_image))
    loss11 = tf.reduce_mean((1. - tf.image.ssim(net_image3.outputs, t_target_image, max_val=1.0)) / 2.)  # SSIM loss
    loss2 = tf.reduce_mean(tf.abs(net_image2.outputs - t_target_image_down1))
    loss3 = tf.reduce_mean(tf.abs(net_image1.outputs - t_target_image_down2))

    mse = alpha[0] * loss1 + alpha[1] * loss2 + alpha[2] * loss3
    # mse = alpha[0] * closs1 + alpha[1] * closs2 + alpha[2] * closs3
    # mse = alpha[0] * loss1
    # 差的平方均值

    test_target_image_down1 = tf.image.resize_images(test_target_image,
                                                     size=[in_patch_size * scale // 2, in_patch_size * scale // 2],
                                                     method=0, align_corners=False)
    # (124, 40, 40, 1)
    test_target_image_down2 = tf.image.resize_images(test_target_image,
                                                     size=[in_patch_size * scale // 4, in_patch_size * scale // 4],
                                                     method=0, align_corners=False)
    # (124, 20, 20, 1)

    tloss1 = tf.reduce_mean(tf.square(net_testimage3.outputs - test_target_image))
    tloss2 = tf.reduce_mean(tf.square(net_testimage2.outputs - test_target_image_down1))
    tloss3 = tf.reduce_mean(tf.square(net_testimage1.outputs - test_target_image_down2))



    test_mse = alpha[0] * tloss1 + alpha[1] * tloss2 + alpha[2] * tloss3
    # test_mse = alpha[0] * tcloss1 + alpha[1] * tcloss2 + alpha[2] * tcloss3
    ##### Loss summary
    mse_loss_sum = tf.summary.scalar("mse_loss", mse)

    test_mse_sum = tf.summary.scalar("test_loss", test_mse)
    # vis_ms(lms)  lms 8各种  取出1 2 4的数据 可见光范围
    # vis_ms(lms)  32 64 64 3
    # tf.clip_by_value(A, 0, 1) 将A中元素都压缩到0-1之间，小于0则设置为0，max同理

    all_sum = tf.summary.merge([mse_loss_sum])

    #########   optimal    Adam or SGD

    t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='HyTestNet')

    # if method == 'Adam':
    lr = tf.placeholder(tf.float32, shape=[])
    g_optim = tf.train.AdamOptimizer(lr, beta1=0.9).minimize(mse, var_list=t_vars)



    ##### GPU setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #### Run the above

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=30)

    with tf.Session() as sess:
        sess.run(init)
        # restore = True
        if restore:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_directory1)
            saver.restore(sess, ckpt.model_checkpoint_path)

        mse_train = []
        mse_valid = []
        t_time = []
        time_s = time.time()



        for i in range(iterations):

            batch_gt, batch_rgb_hp, batch_ms = get_batch(train_gt, train_rgb_hp, train_ms)

            _, mse_loss = sess.run([g_optim, mse],
                                   feed_dict={t_image: batch_ms, rgb_image: batch_rgb_hp, t_target_image: batch_gt,
                                              lr: lr_rate})

            if i % 100 == 0:
                print("Iter: " + str(i) + " MSE: " + str(mse_loss))  # print, e.g.,: Iter: 0 MSE: 0.18406609
                mse_train.append(mse_loss)

            if i % 500 == 0 and i != 0:  # after 1000 iteration, re-set: get_batch
                testbatch_gt, testbatch_rgb_hp, testbatch_ms = get_batch(test_gt, test_rgb_hp, test_ms)
                test_mse_loss, testloss = sess.run([test_mse, tloss1],
                                                   feed_dict={test_image: testbatch_ms, testrgb_image: testbatch_rgb_hp,
                                                              test_target_image: testbatch_gt, lr: lr_rate})
                print("Iter: " + str(i) + " validation_MSE: " + str(test_mse_loss) + '    ' + str(
                    testloss) + '      lr= ' + str(lr_rate))
                mse_valid.append(test_mse_loss)
            if i % 2000 == 0:
                time_e = time.time()
                print(time_e - time_s)
                t_time.append(time_e - time_s)
            if i % 10000 == 0 and i != 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, model_directory + '/model-' + str(i) + '.ckpt')
                print("Save Model")
            if i %10000==0 and i!=0:
                lr_rate = lr_rate / 2



            # if i==180000:
            #     lr_rate=lr_rate/2
        ## finally write the mse info ##
        file = open(model_directory + '/train_mse.txt', 'w')  # write the training error into train_mse.txt
        file.write(str(mse_train))
        file.close()

        file = open(model_directory + '/valid_mse.txt', 'w')  # write the valid error into valid_mse.txt
        file.write(str(mse_valid))
        file.close()
        file = open(model_directory + '/t_time.txt', 'w')  # write the valid error into valid_mse.txt
        file.write(str(t_time))
        file.close()



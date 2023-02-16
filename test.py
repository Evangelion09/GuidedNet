
import warnings
import matplotlib.pyplot as plt
import numpy as np
import cv2
# import tensorflow.contrib.layers as ly
import os
import scipy.io as sio
import time
import h5py
import tensorflow as tf
import tensorlayer as tl
from config import *
from tensorlayer.layers import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



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
    warnings.filterwarnings("ignore")
    batch_size = 32
    res_number=10
    cp=340000 #checkpoint

    test_data = '/Data/Machine Learning/Ran Ran/data/rgbnet/test/test10CSR.mat'

    dataname='output_01.mat'

    model_directory = './models/train_up67_leaky_res10_CSR_RGB124/'

    tf.reset_default_graph()

    test_gt, test_rgb_hp, test_ms, test_lms = get_all_data(test_data)
    print(test_gt.shape)


    print(test_ms.shape)

    # time.sleep(10)

    spectral_bands = test_ms.shape[3]
    N = test_ms.shape[0]

    hh = test_gt.shape[1]  # height
    hw = test_gt.shape[2]  # width
    lh = test_ms.shape[1]
    lw = test_ms.shape[2]
    # print(h)
    # placeholder for tensor
    testgt = tf.placeholder(shape=[1,hh, hw, spectral_bands], dtype=tf.float32)
    testrgb = tf.placeholder(shape=[1, hh, hw, 3], dtype=tf.float32)
    testms = tf.placeholder(shape=[1, lh, lw, spectral_bands], dtype=tf.float32)
    testlms = tf.placeholder(shape=[1,hh, hw, spectral_bands], dtype=tf.float32)


    # net_image3, net_image2, net_image1, = HyTestNet(testms, testrgb, reuse=False)
    net_image3, net_image2,  net_image1= HyTestNet(testms, testrgb, reuse=False)

    output = tf.clip_by_value(net_image3.outputs, 0, 1)  # final output
    output1 = tf.clip_by_value(net_image2.outputs, 0, 1)  # final output
    output2 = tf.clip_by_value(net_image1.outputs, 0, 1)  # final output

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # loading  model
        if tf.train.get_checkpoint_state(model_directory):
            ckpt = tf.train.latest_checkpoint(model_directory)
            ckpt=model_directory+'model-'+str(cp)+'.ckpt'

            # model_checkpoint_path: "model-300000.ckpt"
            saver.restore(sess, ckpt)
            print("load new model")

        else:
            ckpt = tf.train.get_checkpoint_state(model_directory + "pre-trained/")
            saver.restore(sess,
                          ckpt.model_checkpoint_path)  # this model uses 128 feature maps and for debug only
            print("load pre-trained model")



        print('ran')
        print(test_ms.shape)
        print(test_rgb_hp.shape)
        onetest_ms = test_ms[0:1, :, :, :]
        onetest_rgb_hp = test_rgb_hp[0:1, :, :, :]
        final_output = np.zeros(test_gt.shape)
        alltime=0
        for i in range(N):
            onetest_ms[0, :, :, :] = test_ms[i, :, :, :]
            onetest_rgb_hp[0, :, :, :] = test_rgb_hp[i, :, :, :]
            time_s = time.time()
            # [final_temp,final_temp2,final_temp3,o1,o2,o3] = sess.run([output,output1,output2,u11,u22,u33], feed_dict={testms: onetest_ms, testrgb: onetest_rgb_hp})
            final_temp = sess.run(output,feed_dict={testms: onetest_ms, testrgb: onetest_rgb_hp})
            time_e = time.time()
            print(time_e - time_s)
            alltime=alltime+time_e - time_s
            final_output[i,:,:,:]=final_temp

            #     cv2.imshow('7', o2[0, :, :, 20])
            #     cv2.imshow('8', o3[0, :, :, 20])
            #
            #     # img=cv2.resize(img,(1000,1000))
            #     cv2.imshow('1',onetest_ms[0,:,:,20])
            #     tgt = test_gt[i,  :, :, 20]
            #
            #     cv2.imshow('2', tgt)
            #     cv2.waitKey()
                # trgb=np.reshape(onetest_rgb_hp, (3,1000, 1000))
                # trgb = np.transpose(trgb, [1, 2, 0])
                # cv2.imshow('2',trgb)
            #
        print(final_output.shape)
        print(alltime)
        sio.savemat(model_directory+dataname, {'output': final_output})

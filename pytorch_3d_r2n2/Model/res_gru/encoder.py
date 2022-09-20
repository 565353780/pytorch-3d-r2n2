#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn
from torch.nn import \
    Linear, Conv2d, MaxPool2d, LeakyReLU, Tanh, Sigmoid

from pytorch_3d_r2n2.Model.layers import FCConv3DLayer_torch


class Encoder(nn.Module):
    def __init__(self, input_shape, n_convfilter, \
                 n_fc_filters, h_shape, conv3d_filter_shape):
        print("\ninitializing \"Encoder\"")
        #input_shape = (self.batch_size, 3, img_w, img_h)
        super(Encoder, self).__init__()
        #conv1
        self.conv1a = Conv2d(input_shape[1], n_convfilter[0], 7, padding=3)
        self.conv1b = Conv2d(n_convfilter[0], n_convfilter[0], 3, padding=1)

        #conv2
        self.conv2a = Conv2d(n_convfilter[0], n_convfilter[1], 3, padding=1)
        self.conv2b = Conv2d(n_convfilter[1], n_convfilter[1], 3, padding=1)
        self.conv2c = Conv2d(n_convfilter[0], n_convfilter[1], 1)

        #conv3
        self.conv3a = Conv2d(n_convfilter[1], n_convfilter[2], 3, padding=1)
        self.conv3b = Conv2d(n_convfilter[2], n_convfilter[2], 3, padding=1)
        self.conv3c = Conv2d(n_convfilter[1], n_convfilter[2], 1)

        #conv4
        self.conv4a = Conv2d(n_convfilter[2], n_convfilter[3], 3, padding=1)
        self.conv4b = Conv2d(n_convfilter[3], n_convfilter[3], 3, padding=1)

        #conv5
        self.conv5a = Conv2d(n_convfilter[3], n_convfilter[4], 3, padding=1)
        self.conv5b = Conv2d(n_convfilter[4], n_convfilter[4], 3, padding=1)
        self.conv5c = Conv2d(n_convfilter[3], n_convfilter[4], 1)

        #conv6
        self.conv6a = Conv2d(n_convfilter[4], n_convfilter[5], 3, padding=1)
        self.conv6b = Conv2d(n_convfilter[5], n_convfilter[5], 3, padding=1)

        #pooling layer
        self.pool = MaxPool2d(kernel_size=2, padding=1)

        #nonlinearities of the network
        self.leaky_relu = LeakyReLU(negative_slope=0.01)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        #find the input feature map size of the fully connected layer
        fc7_feat_w, fc7_feat_h = self.fc_in_featmap_size(input_shape,
                                                         num_pooling=6)
        #define the fully connected layer
        self.fc7 = Linear(int(n_convfilter[5] * fc7_feat_w * fc7_feat_h),
                          n_fc_filters[0])

        #define the FCConv3DLayers in 3d convolutional gru unit
        self.t_x_s_update = FCConv3DLayer_torch(n_fc_filters[0],
                                                conv3d_filter_shape, h_shape)
        self.t_x_s_reset = FCConv3DLayer_torch(n_fc_filters[0],
                                               conv3d_filter_shape, h_shape)
        self.t_x_rs = FCConv3DLayer_torch(n_fc_filters[0], conv3d_filter_shape,
                                          h_shape)

    def forward(self, x, h, u, time):
        """
        x is the input and the size of x is (batch_size, channels, heights, widths).
        h and u is the hidden state and activation of last time step respectively.
        This function defines the forward pass of the encoder of the network.
        """
        conv1a = self.conv1a(x)
        rect1a = self.leaky_relu(conv1a)
        conv1b = self.conv1b(rect1a)
        rect1 = self.leaky_relu(conv1b)
        pool1 = self.pool(rect1)

        conv2a = self.conv2a(pool1)
        rect2a = self.leaky_relu(conv2a)
        conv2b = self.conv2b(rect2a)
        rect2 = self.leaky_relu(conv2b)
        conv2c = self.conv2c(pool1)
        res2 = conv2c + rect2
        pool2 = self.pool(res2)

        conv3a = self.conv3a(pool2)
        rect3a = self.leaky_relu(conv3a)
        conv3b = self.conv3b(rect3a)
        rect3 = self.leaky_relu(conv3b)
        conv3c = self.conv3c(pool2)
        res3 = conv3c + rect3
        pool3 = self.pool(res3)

        conv4a = self.conv4a(pool3)
        rect4a = self.leaky_relu(conv4a)
        conv4b = self.conv4b(rect4a)
        rect4 = self.leaky_relu(conv4b)
        pool4 = self.pool(rect4)

        conv5a = self.conv5a(pool4)
        rect5a = self.leaky_relu(conv5a)
        conv5b = self.conv5b(rect5a)
        rect5 = self.leaky_relu(conv5b)
        conv5c = self.conv5c(pool4)
        res5 = conv5c + rect5
        pool5 = self.pool(res5)

        conv6a = self.conv6a(pool5)
        rect6a = self.leaky_relu(conv6a)
        conv6b = self.conv6b(rect6a)
        rect6 = self.leaky_relu(conv6b)
        res6 = pool5 + rect6
        pool6 = self.pool(res6)

        pool6 = pool6.view(pool6.size(0), -1)

        fc7 = self.fc7(pool6)
        rect7 = self.leaky_relu(fc7)

        t_x_s_update = self.t_x_s_update(rect7, h)
        t_x_s_reset = self.t_x_s_reset(rect7, h)

        update_gate = self.sigmoid(t_x_s_update)
        complement_update_gate = 1 - update_gate
        reset_gate = self.sigmoid(t_x_s_reset)

        rs = reset_gate * h
        t_x_rs = self.t_x_rs(rect7, rs)
        tanh_t_x_rs = self.tanh(t_x_rs)

        gru_out = update_gate * h + complement_update_gate * tanh_t_x_rs

        return gru_out, update_gate

    #infer the input feature map size, (height, width) of the fully connected layer
    def fc_in_featmap_size(self, input_shape, num_pooling):
        #fully connected layer
        img_w = input_shape[2]
        img_h = input_shape[3]
        #infer the size of the input feature map of the fully connected layer
        fc7_feat_w = img_w
        fc7_feat_h = img_h
        for i in range(num_pooling):
            #image downsampled by pooling layers
            #w_out= np.floor((w_in+ 2*padding[0]- dilation[0]*(kernel_size[0]- 1)- 1)/stride[0]+ 1)
            fc7_feat_w = np.floor((fc7_feat_w + 2 * 1 - 1 * (2 - 1) - 1) / 2 +
                                  1)
            fc7_feat_h = np.floor((fc7_feat_h + 2 * 1 - 1 * (2 - 1) - 1) / 2 +
                                  1)
        return fc7_feat_w, fc7_feat_h

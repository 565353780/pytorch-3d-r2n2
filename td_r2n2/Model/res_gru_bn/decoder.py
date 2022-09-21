#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn import LeakyReLU, Conv3d, BatchNorm3d

from td_r2n2.Model.layers import Unpool3DLayer


class Decoder(nn.Module):

    def __init__(self, n_deconvfilter, h_shape):
        print("initializing \"Decoder\"")
        super(Decoder, self).__init__()
        #3d conv7
        self.conv7a = Conv3d(n_deconvfilter[0],
                             n_deconvfilter[1],
                             3,
                             padding=1)
        self.bn7a = BatchNorm3d(n_deconvfilter[1])
        self.conv7b = Conv3d(n_deconvfilter[1],
                             n_deconvfilter[1],
                             3,
                             padding=1)
        self.bn7b = BatchNorm3d(n_deconvfilter[1])

        #3d conv8
        self.conv8a = Conv3d(n_deconvfilter[1],
                             n_deconvfilter[2],
                             3,
                             padding=1)
        self.bn8a = BatchNorm3d(n_deconvfilter[2])
        self.conv8b = Conv3d(n_deconvfilter[2],
                             n_deconvfilter[2],
                             3,
                             padding=1)
        self.bn8b = BatchNorm3d(n_deconvfilter[2])

        #3d conv9
        self.conv9a = Conv3d(n_deconvfilter[2],
                             n_deconvfilter[3],
                             3,
                             padding=1)
        self.bn9a = BatchNorm3d(n_deconvfilter[3])
        self.conv9b = Conv3d(n_deconvfilter[3],
                             n_deconvfilter[3],
                             3,
                             padding=1)
        self.bn9b = BatchNorm3d(n_deconvfilter[3])
        self.conv9c = Conv3d(n_deconvfilter[2], n_deconvfilter[3], 1)
        self.bn9c = BatchNorm3d(n_deconvfilter[3])

        #3d conv10
        self.conv10a = Conv3d(n_deconvfilter[3],
                              n_deconvfilter[4],
                              3,
                              padding=1)
        self.bn10a = BatchNorm3d(n_deconvfilter[4])
        self.conv10b = Conv3d(n_deconvfilter[4],
                              n_deconvfilter[4],
                              3,
                              padding=1)
        self.bn10b = BatchNorm3d(n_deconvfilter[4])
        self.conv10c = Conv3d(n_deconvfilter[4],
                              n_deconvfilter[4],
                              3,
                              padding=1)
        self.bn10c = BatchNorm3d(n_deconvfilter[4])
        self.conv10d = Conv3d(n_deconvfilter[3], n_deconvfilter[4], 1)
        self.bn10d = BatchNorm3d(n_deconvfilter[4])

        #3d conv11
        self.conv11 = Conv3d(n_deconvfilter[4],
                             n_deconvfilter[5],
                             3,
                             padding=1)

        #unpooling layer
        self.unpool3d = Unpool3DLayer(unpool_size=2)

        #nonlinearities of the network
        self.leaky_relu = LeakyReLU(negative_slope=0.01)

    def forward(self, gru_out):
        unpool7 = self.unpool3d(gru_out)
        conv7a = self.conv7a(unpool7)
        conv7a = self.bn7a(conv7a)
        rect7a = self.leaky_relu(conv7a)
        conv7b = self.conv7b(rect7a)
        conv7b = self.bn7b(conv7b)
        #residual connection before nonlinearity
        res7 = unpool7 + conv7b
        rect7 = self.leaky_relu(res7)

        unpool8 = self.unpool3d(rect7)
        conv8a = self.conv8a(unpool8)
        conv8a = self.bn8a(conv8a)
        rect8a = self.leaky_relu(conv8a)
        conv8b = self.conv8b(rect8a)
        conv8b = self.bn8b(conv8b)
        #residual connection before nonlinearity
        res8 = unpool8 + conv8b
        rect8 = self.leaky_relu(res8)

        unpool9 = self.unpool3d(rect8)
        conv9a = self.conv9a(unpool9)
        conv9a = self.bn9a(conv9a)
        rect9a = self.leaky_relu(conv9a)
        conv9b = self.conv9b(rect9a)
        conv9b = self.bn9b(conv9b)
        conv9c = self.conv9c(unpool9)
        conv9c = self.bn9c(conv9c)
        #residual connection before nonlinearity
        res9 = conv9c + conv9b
        rect9 = self.leaky_relu(res9)

        conv10a = self.conv10a(rect9)
        conv10a = self.bn10a(conv10a)
        rect10a = self.leaky_relu(conv10a)
        conv10b = self.conv10b(rect10a)
        conv10b = self.bn10b(conv10b)
        rect10b = self.leaky_relu(conv10b)
        conv10c = self.conv10c(rect10b)
        #residual connection before nonlinearity
        conv10d = self.conv10d(rect9)
        conv10d = self.bn10d(conv10d)
        res10 = conv10c + conv10d
        rect10 = self.leaky_relu(res10)

        conv11 = self.conv11(rect10)
        return conv11

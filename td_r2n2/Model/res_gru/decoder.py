#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn import LeakyReLU, Conv3d

from td_r2n2.Model.layers import Unpool3DLayer


class Decoder(nn.Module):

    def __init__(self, n_deconvfilter, h_shape):
        super(Decoder, self).__init__()
        #3d conv7
        self.conv7a = Conv3d(n_deconvfilter[0],
                             n_deconvfilter[1],
                             3,
                             padding=1)
        self.conv7b = Conv3d(n_deconvfilter[1],
                             n_deconvfilter[1],
                             3,
                             padding=1)

        #3d conv8
        self.conv8a = Conv3d(n_deconvfilter[1],
                             n_deconvfilter[2],
                             3,
                             padding=1)
        self.conv8b = Conv3d(n_deconvfilter[2],
                             n_deconvfilter[2],
                             3,
                             padding=1)

        #3d conv9
        self.conv9a = Conv3d(n_deconvfilter[2],
                             n_deconvfilter[3],
                             3,
                             padding=1)
        self.conv9b = Conv3d(n_deconvfilter[3],
                             n_deconvfilter[3],
                             3,
                             padding=1)
        self.conv9c = Conv3d(n_deconvfilter[2], n_deconvfilter[3], 1)

        #3d conv10
        self.conv10a = Conv3d(n_deconvfilter[3],
                              n_deconvfilter[4],
                              3,
                              padding=1)
        self.conv10b = Conv3d(n_deconvfilter[4],
                              n_deconvfilter[4],
                              3,
                              padding=1)
        self.conv10c = Conv3d(n_deconvfilter[4],
                              n_deconvfilter[4],
                              3,
                              padding=1)

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
        rect7a = self.leaky_relu(conv7a)
        conv7b = self.conv7b(rect7a)
        rect7 = self.leaky_relu(conv7b)
        res7 = unpool7 + rect7

        unpool8 = self.unpool3d(res7)
        conv8a = self.conv8a(unpool8)
        rect8a = self.leaky_relu(conv8a)
        conv8b = self.conv8b(rect8a)
        rect8 = self.leaky_relu(conv8b)
        res8 = unpool8 + rect8

        unpool9 = self.unpool3d(res8)
        conv9a = self.conv9a(unpool9)
        rect9a = self.leaky_relu(conv9a)
        conv9b = self.conv9b(rect9a)
        rect9 = self.leaky_relu(conv9b)

        conv9c = self.conv9c(unpool9)
        res9 = conv9c + rect9

        conv10a = self.conv10a(res9)
        rect10a = self.leaky_relu(conv10a)
        conv10b = self.conv10b(rect10a)
        rect10 = self.leaky_relu(conv10b)

        conv10c = self.conv10c(rect10)
        res10 = conv10c + rect10

        conv11 = self.conv11(res10)
        return conv11

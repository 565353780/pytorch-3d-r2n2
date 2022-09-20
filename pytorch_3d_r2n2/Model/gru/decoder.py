#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn import LeakyReLU, Conv3d

from pytorch_3d_r2n2.Model.layers import Unpool3DLayer


class decoder(nn.Module):

    def __init__(self, n_deconvfilter, h_shape):
        print("initializing \"decoder\"")
        super(decoder, self).__init__()
        #3d conv7
        conv7_kernel_size = 3
        self.conv7 = Conv3d(in_channels= n_deconvfilter[0], \
                            out_channels= n_deconvfilter[1], \
                            kernel_size= conv7_kernel_size, \
                            padding = int((conv7_kernel_size - 1) / 2))

        #3d conv7
        conv8_kernel_size = 3
        self.conv8 = Conv3d(in_channels= n_deconvfilter[1], \
                            out_channels= n_deconvfilter[2], \
                            kernel_size= conv8_kernel_size, \
                            padding = int((conv8_kernel_size - 1) / 2))

        #3d conv7
        conv9_kernel_size = 3
        self.conv9 = Conv3d(in_channels= n_deconvfilter[2], \
                            out_channels= n_deconvfilter[3], \
                            kernel_size= conv9_kernel_size, \
                            padding = int((conv9_kernel_size - 1) / 2))

        #3d conv7
        conv10_kernel_size = 3
        self.conv10 = Conv3d(in_channels= n_deconvfilter[3], \
                            out_channels= n_deconvfilter[4], \
                            kernel_size= conv10_kernel_size, \
                            padding = int((conv10_kernel_size - 1) / 2))

        #3d conv7
        conv11_kernel_size = 3
        self.conv11 = Conv3d(in_channels= n_deconvfilter[4], \
                            out_channels= n_deconvfilter[5], \
                            kernel_size= conv11_kernel_size, \
                            padding = int((conv11_kernel_size - 1) / 2))

        #pooling layer
        self.unpool3d = Unpool3DLayer(unpool_size=2)

        #nonlinearities of the network
        self.leaky_relu = LeakyReLU(negative_slope=0.01)

    def forward(self, gru_out):
        gru_out_to_conv11 = nn.Sequential(self.unpool3d, self.conv7, self.leaky_relu, \
                                          self.unpool3d, self.conv8, self.leaky_relu, \
                                          self.unpool3d, self.conv9, self.leaky_relu, \
                                          self.conv10, self.leaky_relu, self.conv11)
        return gru_out_to_conv11(gru_out)

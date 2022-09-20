#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import datetime as dt
import torch.nn as nn
from torch.autograd import Variable

from pytorch_3d_r2n2.Config.config import cfg

from pytorch_3d_r2n2.Method.utils import weight_init


class BaseGRUNet(nn.Module):

    def __init__(
        self,
        random_seed=dt.datetime.now().microsecond,
    ):
        super().__init__()
        self.rng = np.random.RandomState(random_seed)

        self.batch_size = cfg.CONST.BATCH_SIZE

        self.img_w = cfg.CONST.IMG_W
        self.img_h = cfg.CONST.IMG_H
        self.n_vox = cfg.CONST.N_VOX

        self.is_x_tensor4 = False

        self.n_gru_vox = 4
        # x.size = (num_views, batch_size, 3, img_w, img_h)
        self.input_shape = (self.batch_size, 3, self.img_w, self.img_h)
        #number of filters for each convolution layer in the encoder
        self.n_convfilter = [96, 128, 256, 256, 256, 256]
        #the dimension of the fully connected layer
        self.n_fc_filters = [1024]
        #number of filters for each 3d convolution layer in the decoder
        self.n_deconvfilter = [128, 128, 128, 64, 32, 2]
        #the size of the hidden state
        self.h_shape = (self.batch_size, self.n_deconvfilter[0],
                        self.n_gru_vox, self.n_gru_vox, self.n_gru_vox)
        #the filter shape of the 3d convolutional gru unit
        self.conv3d_filter_shape = (self.n_deconvfilter[0],
                                    self.n_deconvfilter[0], 3, 3, 3)
        return

    def parameter_init(self):
        #initialize all the parameters of the gru net
        if hasattr(self, "encoder") and hasattr(self, "decoder"):
            for m in self.modules():

                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    """
                    For Conv2d, the shape of the weight is 
                    (out_channels, in_channels, kernel_size[0], kernel_size[1]).
                    For Conv3d, the shape of the weight is 
                    (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]).
                    """
                    w_shape = (m.out_channels, m.in_channels, *m.kernel_size)
                    m.weight.data = weight_init(w_shape)
                    if m.bias is not None:
                        m.bias.data.fill_(0.1)

                elif isinstance(m, nn.Linear):
                    """
                    For Linear module, the shape of the weight is (out_features, in_features)
                    """
                    w_shape = (m.out_features, m.in_features)
                    m.weight.data = weight_init(w_shape)
                    if m.bias is not None:
                        m.bias.data.fill_(0.1)
        else:
            raise Exception(
                "The network must have an encoder and a decoder before initializing all the parameters"
            )

    def initHidden(self, h_shape):
        h = torch.zeros(h_shape).cuda()
        return Variable(h)

    def forward(self, data):
        # x.size = (num_views, batch_size, channels, heights, widths)
        h = self.initHidden(self.h_shape)
        u = self.initHidden(self.h_shape)

        # store intermediate update gate activations
        data['predictions']['activations'] = []

        for time in range(data['inputs']['images'].size(0)):
            gru_out, update_gate = self.encoder(data['inputs']['images'][time],
                                                h, u, time)

            h = gru_out
            u = update_gate
            data['predictions']['activations'].append(u)

        data['predictions']['decoder_out'] = self.decoder(h)

        max_channel = torch.max(data['predictions']['decoder_out'],
                                dim=1,
                                keepdim=True)[0]
        data['predictions'][
            'adj_inputs'] = data['predictions']['decoder_out'] - max_channel

        exp_x = torch.exp(data['predictions']['adj_inputs'])
        data['predictions']['sum_exp_x'] = torch.sum(exp_x,
                                                     dim=1,
                                                     keepdim=True)
        data['predictions'][
            'prediction'] = exp_x / data['predictions']['sum_exp_x']

        if self.training:
            data = self.loss(data)
        return data

    def loss(self, data):
        #the size of images and voxels is (batch_size, channels, depth, height, width)
        #torch.max return a tuple of (max_value, index_of_max_value)
        assert data['inputs']['voxels'] is not None
        data['losses']['loss'] = torch.mean(
            torch.sum(-data['inputs']['voxels'] *
                      data['predictions']['adj_inputs'],
                      dim=1,
                      keepdim=True) +
            torch.log(data['predictions']['sum_exp_x']))
        return data

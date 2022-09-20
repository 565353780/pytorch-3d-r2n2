#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pytorch_3d_r2n2.Model.base_gru.base_gru_net import BaseGRUNet
from pytorch_3d_r2n2.Model.gru_net.encoder import Encoder
from pytorch_3d_r2n2.Model.gru_net.decoder import Decoder


class GRUNet(BaseGRUNet):

    def __init__(self):
        print("initializing \"GRUNet\"")
        super(GRUNet, self).__init__()
        """
        Set the necessary data of the network
        """

        #set the encoder and the decoder of the network
        self.encoder = Encoder(self.input_shape, self.n_convfilter, \
                               self.n_fc_filters, self.h_shape, self.conv3d_filter_shape)

        self.decoder = Decoder(self.n_deconvfilter, self.h_shape)

        #initialize all the parameters
        self.parameter_init()

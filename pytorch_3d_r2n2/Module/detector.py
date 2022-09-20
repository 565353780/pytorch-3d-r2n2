#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image

from pytorch_3d_r2n2.Config.config import cfg

from pytorch_3d_r2n2.Model.res_gru.res_gru_net import ResidualGRUNet

from pytorch_3d_r2n2.Method.augment import preprocess_img


class Detector(object):

    def __init__(self, model_file_path=None):
        self.model = ResidualGRUNet()
        self.lr = cfg.TRAIN.DEFAULT_LEARNING_RATE

        self.checkpoint = None

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model_file not exist!")
            return False

        self.model.cuda()
        self.model.eval()

        print("[INFO][Detector]")
        print("\t start loading checkpoint from " + model_file_path + "...")
        self.checkpoint = torch.load(model_file_path)

        net_state = self.checkpoint['net_state']
        self.model.load_state_dict(net_state)
        return True

    def detectImages(self, image_list):
        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        valid_image_list = []
        for image in image_list:
            valid_image = image.resize((cfg.CONST.IMG_H, cfg.CONST.IMG_W),
                                       Image.ANTIALIAS)
            valid_image = preprocess_img(valid_image, train=False)
            valid_image = np.array(valid_image).transpose(
                (2, 0, 1)).astype(np.float32)
            valid_image_list.append([valid_image])
        valid_image_array = np.array(valid_image_list, dtype=np.float32)

        data['inputs']['images'] = torch.from_numpy(valid_image_array).cuda()
        data['inputs']['voxels'] = None

        data = self.model(data)
        return data

    def detectImageFiles(self, image_file_path_list):
        image_list = []
        for image_file_path in image_file_path_list:
            image = Image.open(image_file_path)
            image_list.append(image)

        data = self.detectImages(image_list)
        return data


def demo():
    return True

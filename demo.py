#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image

from pytorch_3d_r2n2.Config.config import cfg, cfg_from_list

from pytorch_3d_r2n2.Model.res_gru.res_gru_net import ResidualGRUNet

from pytorch_3d_r2n2.Method.augment import preprocess_img
from pytorch_3d_r2n2.Method.voxel import voxel2obj

from pytorch_3d_r2n2.Module.trainer import Solver

DEFAULT_WEIGHTS = '/home/chli/chLi/3D-R2N2/checkpoint.pth'


def load_demo_images():
    img_h = cfg.CONST.IMG_H
    img_w = cfg.CONST.IMG_W

    image_list = []

    for i in range(3):
        image = Image.open('./images/%d.png' % i)
        image = image.resize((img_h, img_w), Image.ANTIALIAS)
        image = preprocess_img(image, train=False)
        image_list.append([np.array(image).transpose( \
                        (2, 0, 1)).astype(np.float32)])
    image_array = np.array(image_list).astype(np.float32)
    return torch.from_numpy(image_array)


def main():
    pred_file_name = 'prediction.obj'

    demo_imgs = load_demo_images()

    net = ResidualGRUNet()

    net.cuda()

    net.eval()

    solver = Solver(net)
    solver.load(DEFAULT_WEIGHTS)

    # Run the network
    voxel_prediction, _ = solver.test_output(demo_imgs)
    voxel_prediction = voxel_prediction.detach().cpu().numpy()

    # Save the prediction to an OBJ file (mesh file).
    voxel2obj(pred_file_name, voxel_prediction[0, 1] > cfg.TEST.VOXEL_THRESH)

    return


if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])

    main()

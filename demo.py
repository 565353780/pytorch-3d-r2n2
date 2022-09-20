#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import numpy as np

import torch

from PIL import Image

from pytorch_3d_r2n2.Model.res_gru.res_gru_net import ResidualGRUNet
from pytorch_3d_r2n2.lib.config import cfg, cfg_from_list
from pytorch_3d_r2n2.lib.data_augmentation import preprocess_img
from pytorch_3d_r2n2.lib.solver import Solver
from pytorch_3d_r2n2.lib.voxel import voxel2obj

DEFAULT_WEIGHTS = '/home/chli/chLi/3D-R2N2/checkpoint.pth'

def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def load_demo_images():
    img_h = cfg.CONST.IMG_H
    img_w = cfg.CONST.IMG_W

    imgs = []

    for i in range(3):
        img = Image.open('./pytorch_3d_r2n2/imgs/%d.png' % i)
        img = img.resize((img_h, img_w), Image.ANTIALIAS)
        img = preprocess_img(img, train=False)
        imgs.append([np.array(img).transpose( \
                        (2, 0, 1)).astype(np.float32)])
    ims_np = np.array(imgs).astype(np.float32)
    return torch.from_numpy(ims_np)


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

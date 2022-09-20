#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pytorch_3d_r2n2.Config.config import cfg, cfg_from_list

from pytorch_3d_r2n2.Model.res_gru.res_gru_net import ResidualGRUNet

from pytorch_3d_r2n2.Method.voxel import voxel2obj

from pytorch_3d_r2n2.Module.trainer import Trainer


def demo():
    net = ResidualGRUNet()

    net.cuda()

    net.eval()

    solver = Trainer(net)
    solver.load('/home/chli/chLi/3D-R2N2/checkpoint.pth')

    # Run the network
    image_file_path_list = []
    for i in range(3):
        image_file_path = "./images/" + str(i) + ".png"
        image_file_path_list.append(image_file_path)
    save_obj_path = "./prediction.obj"
    data = solver.detectImageFiles(image_file_path_list)
    voxel_prediction = data['predictions']['prediction'].detach().cpu().numpy()

    voxel2obj(save_obj_path, voxel_prediction[0, 1] > cfg.TEST.VOXEL_THRESH)
    return True


if __name__ == '__main__':
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    demo()

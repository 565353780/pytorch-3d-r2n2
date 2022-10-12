#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from td_r2n2.Config.config import cfg_from_list

from td_r2n2.Module.detector import TDR2N2Detector


def demo():
    model_file_path = "/home/chli/chLi/3D-R2N2/checkpoint.pth"
    #  model_file_path = "./output/models/checkpoint.pth"

    #  image_folder_path = "./images/"
    image_folder_path = "/home/chli/chLi/3D-R2N2/test_images/screen/white/"

    save_obj_file_path = "/home/chli/chLi/3D-R2N2/test_images/screen/screen.obj"

    cfg_from_list(['CONST.BATCH_SIZE', 1])

    td_r2n2_detector = TDR2N2Detector(model_file_path)

    image_file_path_list = []
    image_file_name_list = os.listdir(image_folder_path)
    for image_file_name in image_file_name_list:
        image_file_path = image_folder_path + image_file_name
        image_file_path_list.append(image_file_path)
    data = td_r2n2_detector.detectImageFiles(image_file_path_list)
    td_r2n2_detector.saveAsObj(data, save_obj_file_path)
    return True

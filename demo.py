#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pytorch_3d_r2n2.Config.config import cfg_from_list

from pytorch_3d_r2n2.Module.td_r2n2_detector import demo

if __name__ == '__main__':
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    demo()

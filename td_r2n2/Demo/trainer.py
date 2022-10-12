#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader

from td_r2n2.Config.config import cfg
from td_r2n2.Dataset.dataset import ShapeNetCollateFn, ShapeNetDataset

from td_r2n2.Module.trainer import cleanup_handle, TDR2N2Trainer


@cleanup_handle
def demo():
    #  cfg.DATASET = '/cvgl/group/ShapeNet/ShapeNetCore.v1/cat1000.json'
    cfg.CONST.RECNET = 'rec_net'
    cfg.TRAIN.DATASET_PORTION = [0, 0.8]

    train_dataset = ShapeNetDataset(cfg.TRAIN.DATASET_PORTION)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.CONST.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.TRAIN.NUM_WORKER,
                              collate_fn=ShapeNetCollateFn(),
                              pin_memory=True)

    val_dataset = ShapeNetDataset(cfg.TEST.DATASET_PORTION)
    val_collate_fn = ShapeNetCollateFn(train=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfg.CONST.BATCH_SIZE,
                            shuffle=True,
                            num_workers=1,
                            collate_fn=val_collate_fn,
                            pin_memory=True)

    model_file_path = "./output/models/checkpoint.pth"

    td_r2n2_trainer = TDR2N2Trainer(model_file_path)
    td_r2n2_trainer.train(train_loader, val_loader)
    return True

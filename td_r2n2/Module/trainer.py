#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from td_r2n2.Config.config import cfg
from td_r2n2.Dataset.dataset import ShapeNetCollateFn, ShapeNetDataset
from td_r2n2.Method.time import getCurrentTimeStr
from td_r2n2.Method.utils import has_nan, max_or_nan
from td_r2n2.Module.detector import TDR2N2Detector

#  from td_r2n2.Method.process import kill_processes


class TDR2N2Trainer(TDR2N2Detector):

    def __init__(self, model_file_path=None):
        self.optimizer = None
        self.scheduler = None
        self.start_train_step = 0

        super().__init__(model_file_path)

        log_folder_path = cfg.LOG.log_folder_path + getCurrentTimeStr() + "/"
        os.makedirs(log_folder_path, exist_ok=True)
        self.writer = SummaryWriter(log_folder_path)

        self.initOptimizer(cfg.TRAIN.POLICY)
        self.initScheduler(cfg.TRAIN.LEARNING_RATES)
        return

    def loadModel(self, model_file_path):
        if not super().loadModel(model_file_path):
            print("[ERROR][TDR2N2Trainer::loadModel]")
            print("\t loadModel in TDR2N2Detector failed!")
            return False
        self.model.train()

        self.initOptimizer(cfg.TRAIN.POLICY)
        self.initScheduler(cfg.TRAIN.LEARNING_RATES)

        if not self.loadOptimizer():
            print("[ERROR][TDR2N2Trainer::loadModel]")
            print("\t loadOptimizer failed!")
            return False

        if 'train_step' in self.checkpoint.keys():
            self.start_train_step = self.checkpoint['train_step']
            print("[INFO][TDR2N2Trainer::loadModel]")
            print("\t start training from train_step = " +
                  str(self.start_train_step) + "...")
        return True

    def initOptimizer(self, policy):
        if self.optimizer is not None:
            return True

        assert policy in ['adam', 'sgd']

        if policy == 'adam':
            self.optimizer = Adam(self.model.parameters(),
                                  lr=self.lr,
                                  weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        else:
            momentum = cfg.TRAIN.MOMENTUM
            self.optimizer = SGD(self.model.parameters(),
                                 lr=self.lr,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 momentum=momentum)
        return True

    def initScheduler(self, lr_dict):
        if self.scheduler is not None:
            return True

        lr_steps = [int(k) for k in lr_dict.keys()]
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                  lr_steps,
                                                  gamma=0.1)
        return True

    def loadOptimizer(self):
        if self.checkpoint is not None:
            optim_state = self.checkpoint['optimizer_state']
            self.optimizer.load_state_dict(optim_state)
        return True

    def saveModel(self, step):
        save_dir = cfg.DIR.OUT_PATH
        os.makedirs(save_dir, exist_ok=True)

        save_path = save_dir + 'checkpoint' + str(step) + '.pth'

        state_dict = {
            'net_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_step': step,
        }
        torch.save(state_dict, save_path)

        symlink_path = save_dir + 'checkpoint.pth'
        if os.path.lexists(symlink_path):
            os.remove(symlink_path)

        os.symlink(os.path.abspath(save_path), os.path.abspath(symlink_path))
        return True

    def trainStep(self, images, voxels):
        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}
        data['inputs']['images'] = images.cuda()
        data['inputs']['voxels'] = voxels.cuda()

        data = self.model(data)

        self.optimizer.zero_grad()
        data['losses']['loss'].backward()
        self.optimizer.step()
        return data

    def valStep(self, val_loader):
        val_losses = 0
        val_num_iter = min(cfg.TRAIN.NUM_VALIDATION_ITERATIONS,
                           len(val_loader))
        val_loader_iter = iter(val_loader)
        for _ in range(val_num_iter):
            batch_img, batch_voxel = val_loader_iter.next()

            data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}
            data['inputs']['images'] = batch_img.cuda()
            data['inputs']['voxels'] = batch_voxel.cuda()

            data = self.model(data)

            val_loss = data['losses']['loss']
            val_losses += val_loss
        val_losses_mean = val_losses / val_num_iter
        return val_losses_mean

    def train(self, train_loader, val_loader=None):
        train_loader_iter = iter(train_loader)
        for train_step in tqdm(
                range(self.start_train_step, cfg.TRAIN.NUM_ITERATION + 1)):
            try:
                batch_img, batch_voxel = train_loader_iter.next()
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch_img, batch_voxel = train_loader_iter.next()

            data = self.trainStep(batch_img, batch_voxel)
            self.scheduler.step()
            loss = data['losses']['loss']
            self.writer.add_scalar("train/loss", loss, train_step)
            self.writer.add_scalar("train/lr",
                                   self.optimizer.param_groups[0]['lr'],
                                   train_step)

            if train_step % cfg.TRAIN.PRINT_FREQ == 0:
                print('%s Iter: %d Loss: %f' %
                      (datetime.now(), train_step, loss))

            if train_step % cfg.TRAIN.VALIDATION_FREQ == 0 and val_loader is not None:
                val_losses_mean = self.valStep(val_loader)
                self.writer.add_scalar("val/loss", val_losses_mean, train_step)
                print('%s Test loss: %f' % (datetime.now(), val_losses_mean))

            if train_step % cfg.TRAIN.NAN_CHECK_FREQ == 0:
                nan_or_max_param = max_or_nan(self.model.parameters())
                if has_nan(nan_or_max_param):
                    print("[ERROR][TDR2N2Trainer::train]")
                    print("\t NaN detected! stop training!")
                    return False

            if (train_step % cfg.TRAIN.SAVE_FREQ == 0 and not train_step == 0) or \
                    (train_step == cfg.TRAIN.NUM_ITERATION):
                self.saveModel(train_step)
        return True


def cleanup_handle(func):
    '''Cleanup the data processes before exiting the program'''

    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print('Wait until the dataprocesses to end')
            # kill_processes(train_queue, train_processes)
            # kill_processes(val_queue, val_processes)
            raise

    return func_wrapper


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

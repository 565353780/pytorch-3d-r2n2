#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pytorch_3d_r2n2.Config.config import cfg

from pytorch_3d_r2n2.Method.time import getCurrentTimeStr
from pytorch_3d_r2n2.Method.utils import has_nan, max_or_nan

from pytorch_3d_r2n2.Module.detector import Detector


class Trainer(Detector):

    def __init__(self, model_file_path=None):
        super().__init__(model_file_path)
        self.start_train_step = 0
        self.optimizer = None
        self.lr_scheduler = None

        log_folder_path = cfg.LOG.log_folder_path + getCurrentTimeStr() + "/"
        os.makedirs(log_folder_path, exist_ok=True)
        self.writer = SummaryWriter(log_folder_path)
        return

    def loadOptimizer(self, policy):
        assert policy in ['adam', 'sgd']

        if policy == 'adam':
            self.optimizer = Adam(self.model.parameters(),
                                  lr=self.lr,
                                  weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        else
            momentum = cfg.TRAIN.MOMENTUM
            self.optimizer = SGD(self.model.parameters(),
                                 lr=self.lr,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 momentum=momentum)

        if self.checkpoint is not None:
            optim_state = self.checkpoint['optimizer_state']
            self.optimizer.load_state_dict(optim_state)
        return True

    def loadModel(self, model_file_path):
        if not super().loadModel(model_file_path):
            print("[ERROR][Trainer::loadModel]")
            print("\t loadModel in Detector failed!")
            return False

        self.model.train()
        if not self.loadOptimizer(cfg.TRAIN.POLICY):
            print("[ERROR][Trainer::loadModel]")
            print("\t loadOptimizer failed!")
            return False

        self.start_train_step = self.checkpoint['train_step']
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
        val_num_iter = min(cfg.TRAIN.NUM_VALIDATION_ITERATIONS, len(val_loader))
        val_loader_iter = iter(val_loader)
        for _ in range(val_num_iter):
            batch_img, batch_voxel = val_loader_iter.next()
            data = self.trainStep(batch_img, batch_voxel)
            val_loss = data['losses']['loss']
            val_losses += val_loss
        val_losses_mean = val_losses / val_num_iter
        return val_losses_mean

    def train(self, train_loader, val_loader=None):
        training_losses = []

        lr_steps = [int(k) for k in cfg.TRAIN.LEARNING_RATES.keys()]

        #Setup the lr_scheduler
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                     lr_steps,
                                                     gamma=0.1)

        train_loader_iter = iter(train_loader)
        for train_step in tqdm(range(start_train_step, cfg.TRAIN.NUM_ITERATION + 1)):
            self.lr_scheduler.step()

            try:
                batch_img, batch_voxel = train_loader_iter.next()
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch_img, batch_voxel = train_loader_iter.next()

            data = self.trainStep(batch_img, batch_voxel)
            loss = data['losses']['loss']
            self.writer.add_scalar("train/loss", loss, train_step)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], train_step)

            training_losses.append(loss.item())

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
                    print("[ERROR][Trainer::train]")
                    print("\t NaN detected! stop training!")
                    return False

            if (train_step % cfg.TRAIN.SAVE_FREQ == 0 and not train_step == 0) or \
                    (train_step == cfg.TRAIN.NUM_ITERATION):
                self.writer.add_scalar("train/loss", training_losses, train_step)
                self.saveModel(train_step)

            #loss is a Variable containing torch.FloatTensor of size 1
            if loss.item() > cfg.TRAIN.LOSS_LIMIT:
                print("[ERROR][Trainer::train]")
                print("\t cost exceeds the threshold! stop training!")
                return False
        return True

def demo():
    return True

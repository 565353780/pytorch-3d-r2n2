#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader


from pytorch_3d_r2n2.Config.config import cfg

from pytorch_3d_r2n2.Model.res_gru.res_gru_net import ResidualGRUNet

from pytorch_3d_r2n2.Dataset.dataset import ShapeNetDataset, ShapeNetCollateFn

#  from pytorch_3d_r2n2.Method.process import kill_processes

from pytorch_3d_r2n2.Module.trainer import Solver

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
def train_net():
    net = ResidualGRUNet()
    print('\nNetwork definition: ')
    print(net)

    if net.is_x_tensor4 and cfg.CONST.N_VIEWS > 1:
        raise ValueError('Do not set the config.CONST.N_VIEWS > 1 when using' \
                         'single-view reconstruction network')

    train_dataset = ShapeNetDataset(cfg.TRAIN.DATASET_PORTION)
    train_collate_fn = ShapeNetCollateFn()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.CONST.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.TRAIN.NUM_WORKER,
                              collate_fn=train_collate_fn,
                              pin_memory=True)

    val_dataset = ShapeNetDataset(cfg.TEST.DATASET_PORTION)
    val_collate_fn = ShapeNetCollateFn(train=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfg.CONST.BATCH_SIZE,
                            shuffle=True,
                            num_workers=1,
                            collate_fn=val_collate_fn,
                            pin_memory=True)

    net.cuda()

    # Generate the solver
    solver = Solver(net)

    # Train the network
    solver.train(train_loader, val_loader)


def main():
    '''Test function'''
    cfg.DATASET = '/cvgl/group/ShapeNet/ShapeNetCore.v1/cat1000.json'
    cfg.CONST.RECNET = 'rec_net'
    cfg.TRAIN.DATASET_PORTION = [0, 0.8]
    train_net()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import sklearn.metrics
from torch.utils.data import DataLoader

from pytorch_3d_r2n2.Config.config import cfg

from pytorch_3d_r2n2.Model.res_gru.res_gru_net import ResidualGRUNet

from pytorch_3d_r2n2.Dataset.dataset import ShapeNetDataset, ShapeNetCollateFn

from pytorch_3d_r2n2.Method.voxel import evaluate_voxel_prediction

from pytorch_3d_r2n2.Module.trainer import Trainer


def test_net():
    result_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.TEST.EXP_NAME)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_fn = os.path.join(result_dir, 'result.mat')

    print("Exp file will be written to: " + result_fn)

    net = ResidualGRUNet()
    net.cuda()

    net.eval()

    solver = Trainer(net)
    solver.load(cfg.CONST.WEIGHTS)

    # set constants
    batch_size = cfg.CONST.BATCH_SIZE

    # set up testing data process. We make only one prefetching process. The
    # process will return one batch at a time.

    test_dataset = ShapeNetDataset(dataset_portion=cfg.TEST.DATASET_PORTION)
    test_collate_fn = ShapeNetCollateFn(train=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=test_collate_fn,
                             pin_memory=True)

    num_data = len(test_dataset)
    num_batch = int(num_data / batch_size)

    # prepare result container
    results = {
        'cost': np.zeros(num_batch),
        'mAP': np.zeros((num_batch, batch_size))
    }
    # Save results for various thresholds
    for thresh in cfg.TEST.VOXEL_THRESH:
        results[str(thresh)] = np.zeros((num_batch, batch_size, 5))

    # Get all test data
    batch_idx = 0
    for batch_img, batch_voxel in test_loader:
        if batch_idx == num_batch:
            break

        data = {
            'inputs': {},
            'predictions': {},
            'losses': {},
            'logs': {}
        }

        data['inputs']['images'] = batch_img
        data['inputs']['voxels'] = batch_voxel

        #activations is a list of torch.cuda.FloatTensor
        pred, loss, activations = solver.detect(data)

        #convert pytorch tensor to numpy array
        pred = pred.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        batch_voxel_np = batch_voxel.cpu().numpy()

        for j in range(batch_size):
            # Save IoU per thresh
            for i, thresh in enumerate(cfg.TEST.VOXEL_THRESH):
                r = evaluate_voxel_prediction(pred[j, ...],
                                              batch_voxel_np[j, ...], thresh)
                results[str(thresh)][batch_idx, j, :] = r

            # Compute AP
            precision = sklearn.metrics.average_precision_score(
                batch_voxel[j, 1].flatten(), pred[j, 1].flatten())

            results['mAP'][batch_idx, j] = precision

        # record result for the batch
        results['cost'][batch_idx] = float(loss)
        print('%d/%d, costs: %f, mAP: %f' %
              (batch_idx, num_batch, loss, np.mean(results['mAP'][batch_idx])))
        batch_idx += 1

    print('Total loss: %f' % np.mean(results['cost']))
    print('Total mAP: %f' % np.mean(results['mAP']))

    # sio.savemat(result_fn, results)

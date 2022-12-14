#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from collections import OrderedDict

from td_r2n2.Config.config import cfg


def id_to_name(id, category_list):
    for k, v in category_list.items():
        if v[0] <= id and v[1] > id:
            return (k, id - v[0])


def category_model_id_pair(dataset_portion=[]):
    '''
    Load category, model names from a shapenet dataset.
    '''

    def model_names(model_path):
        """ Return model names"""
        model_names = [
            name for name in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, name))
        ]
        return sorted(model_names)

    category_name_pair = []  # full path of the objs files

    cats = json.load(open(cfg.DATASET))
    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))
    for _, cat in cats.items():  # load by categories
        model_path = os.path.join(cfg.DIR.SHAPENET_QUERY_PATH, cat['id'])
        # category = cat['name']
        models = model_names(model_path)
        num_models = len(models)

        portioned_models = models[int(num_models * dataset_portion[0]
                                      ):int(num_models * dataset_portion[1])]

        category_name_pair.extend([(cat['id'], model_id)
                                   for model_id in portioned_models])

    print("[INFO][io::category_model_id_pair]")
    print("\t model paths from " + cfg.DATASET)

    return category_name_pair


def get_model_file(category, model_id):
    return cfg.DIR.MODEL_PATH % (category, model_id)


def get_voxel_file(category, model_id):
    return cfg.DIR.VOXEL_PATH % (category, model_id)


def get_rendering_file(category, model_id, rendering_id):
    return os.path.join(cfg.DIR.RENDERING_PATH % (category, model_id),
                        '%02d.png' % rendering_id)


def category_model_id_dict(dataset_portion=[]):
    '''
    Load category, model names from a shapenet dataset.
    '''

    def model_names(model_path):
        """ Return model names"""
        model_names = [
            name for name in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, name))
        ]
        return sorted(model_names)

    category_name_dict_list = []  # full path of the objs files

    cats = json.load(open(cfg.DATASET))
    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))

    counter = 0

    for _, cat in cats.items():  # load by categories
        model_path = os.path.join(cfg.DIR.SHAPENET_QUERY_PATH, cat['id'])
        # category = cat['name']
        models = model_names(model_path)

        num_models = len(models)

        portioned_models_range = (int(num_models * dataset_portion[0]),\
                                  int(num_models * dataset_portion[1]))

        portioned_models = models[
            portioned_models_range[0]:portioned_models_range[1]]

        num_portioned_models = len(portioned_models)


        category_name_dict_list.append({'category_id':cat['id'],\
                                        'category_name':cat['name'],\
                                        'portioned_model_ids':portioned_models,\
                                        'num_portioned_models':num_models,\
                                        'range_in_test': (counter, counter+num_portioned_models)})
        counter += num_portioned_models
    print("[INFO][io::category_model_id_dict]")
    print("\t model paths from " + cfg.DATASET)

    return category_name_dict_list

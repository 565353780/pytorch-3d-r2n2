#!/usr/bin/env python
# -*- coding: utf-8 -*-

from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATASET = './experiments/dataset/shapenet_1000.json'  # yaml/json file that specifies a dataset (training/testing)
__C.NET_NAME = 'res_gru_net'
__C.PROFILE = False
__C.QUEUE_SIZE = 15  # maximum number of minibatches that can be put in a data queue

CONST = edict()

CONST.DEVICE = 'cuda0'
CONST.RNG_SEED = 0
CONST.IMG_W = 127
CONST.IMG_H = 127
CONST.N_VOX = 32
CONST.N_VIEWS = 5
CONST.BATCH_SIZE = 36
CONST.NETWORK_CLASS = 'ResidualGRUNet'
CONST.WEIGHTS = ''  # when set, load the weights from the file

__C.CONST = CONST

DIR = edict()
DIR.SHAPENET_QUERY_PATH = '/home/chli/chLi/3D-R2N2/ShapeNetVox32/'
DIR.MODEL_PATH = '/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v1/%s/%s/model.obj'
DIR.VOXEL_PATH = '/home/chli/chLi/3D-R2N2/ShapeNetVox32/%s/%s/model.binvox'
DIR.RENDERING_PATH = '/home/chli/chLi/3D-R2N2/ShapeNetRendering/%s/%s/rendering'
DIR.OUT_PATH = './output/models/'

__C.DIR = DIR

TRAIN = edict()
TRAIN.RESUME_TRAIN = False
TRAIN.INITIAL_ITERATION = 0  # when the training resumes, set the iteration number
TRAIN.USE_REAL_IMG = False
TRAIN.DATASET_PORTION = [0, 0.8]

TRAIN.NUM_WORKER = 5  # number of data workers
TRAIN.NUM_ITERATION = 60000  # maximum number of training iterations
TRAIN.WORKER_LIFESPAN = 100  # if use blender, kill a worker after some iteration to clear cache
TRAIN.WORKER_CAPACITY = 1000  # if use OSG, load only limited number of models at a time
TRAIN.NUM_RENDERING = 24
TRAIN.NUM_VALIDATION_ITERATIONS = 24
TRAIN.VALIDATION_FREQ = 2000
TRAIN.NAN_CHECK_FREQ = 2000
TRAIN.RANDOM_NUM_VIEWS = True  # feed in random # views if n_views > 1

TRAIN.RANDOM_CROP = True
TRAIN.PAD_X = 10
TRAIN.PAD_Y = 10
TRAIN.FLIP = True

# For no random bg images, add random colors
TRAIN.NO_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
TRAIN.RANDOM_BACKGROUND = False
TRAIN.SIMPLE_BACKGROUND_RATIO = 0.5  # ratio of the simple backgrounded images

# Learning
# For SGD use 0.1, for ADAM, use 0.0001
TRAIN.DEFAULT_LEARNING_RATE = 1e-4
TRAIN.POLICY = 'adam'  # def: sgd, adam
# The EasyDict can't use dict with integers as keys
TRAIN.LEARNING_RATES = {'20000': 1e-5, '60000': 1e-6}
TRAIN.MOMENTUM = 0.90
# weight decay or regularization constant. If not set, the loss can diverge
# after the training almost converged since weight can increase indefinitely
# (for cross entropy loss). Too high regularization will also hinder training.
TRAIN.WEIGHT_DECAY = 0.00005
TRAIN.LOSS_LIMIT = 2  # stop training if the loss exceeds the limit
TRAIN.SAVE_FREQ = 10000  # weights will be overwritten every save_freq
TRAIN.PRINT_FREQ = 40

__C.TRAIN = TRAIN

TEST = edict()

TEST.EXP_NAME = 'test'
TEST.USE_IMG = False
TEST.MODEL_ID = []
TEST.DATASET_PORTION = [0.8, 1]
TEST.SAMPLE_SIZE = 0
TEST.IMG_PATH = ''
TEST.AZIMUTH = []
TEST.NO_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]

TEST.VISUALIZE = False
TEST.VOXEL_THRESH = [0.4]

__C.TEST = TEST

LOG = edict()

LOG.log_folder_path = "./output/logs/"

__C.LOG = LOG

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b.keys():
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(
                ('Type mismatch ({} vs. {}) '
                 'for config key: {}').format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

from pytorch_lightning.utilities.parsing import AttributeDict
import argparse
import hashlib
import pickle
import skimage.transform
import numpy as np
import torch
import os
import pandas as pd

import random
import yaml

# Load the YAML file
with open('training_configs/cardiac_rbaca.yml', 'r') as file:

    config = yaml.safe_load(file)

# Extract the seed value
SEED = config['trainparams']['seed']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # If true, the training speed might be slower.

set_seed(SEED)


def default_params(dparams, params):
    """Copies all key value pairs from params to dparams if not present"""
    matched_params = dparams.copy()
    default_keys = dparams.keys()
    param_keys = params.keys()
    for key in param_keys:
        matched_params[key] = params[key]
        if key in default_keys:
            if (type(params[key]) is dict) and (type(dparams[key]) is dict):
                matched_params[key] = default_params(dparams[key], params[key])
    return matched_params


def sort_dict(input_dict):
    dict_out = {}
    keys = list(input_dict.keys())
    keys.sort()
    for key in keys:
        if type(input_dict[key]) is dict:
            value = sort_dict(input_dict[key])
        else:
            value = input_dict[key]
        dict_out[key] = value
    return dict_out

def get_expname(hparams):
    if type(hparams) is argparse.Namespace:
        hparams = vars(hparams).copy()
    elif type(hparams) is AttributeDict:
        hparams = dict(hparams)

    hashed_params = hash(hparams, length=10)

    expname = hparams['task']
    expname += '_cont' if hparams['continuous'] else '_batch'

    if 'naive_continuous' in hparams:
        expname += '_naive'

    expname += '_' + os.path.splitext(os.path.basename(hparams['datasetfile']))[0]
    if hparams['base_model']:
        expname += '_basemodel_' + hparams['base_model'].split('_')[1]
    if hparams['continuous']:
        expname += '_memory' if hparams['use_memory'] else '_nomemory'
        expname += '_tf{}'.format(str(hparams['transition_phase_after']).replace('.', ''))
    else:
        expname += '_' + '-'.join(hparams['noncontinuous_train_splits'])
    expname += '_'+str(hparams['run_postfix'])
    expname += '_'+hashed_params
    return expname


def save_memory_to_csv(memory, savepath):
    df_cache = pd.DataFrame({'filepath':[ci.filepath for ci in memory],
                             'scanner': [ci.scanner for ci in memory],
                             'pseudo_domain':  [ci.pseudo_domain for ci in memory],
                             'traincounter': [ci.traincounter for ci in memory]})
    df_cache.to_csv(savepath, index=False, index_label=False)

def hash(item, length=40):
    assert (type(item) is dict)
    item = sort_dict(item)
    return hashlib.sha1(pickle.dumps(item)).hexdigest()[0:length]

def resize(img, size, order=1, anti_aliasing=True):
    for i in range(len(size)):
        if size[i] is None:
            size[i] = img.shape[i]
    return skimage.transform.resize(img, size, order=order, mode='reflect', anti_aliasing=anti_aliasing, preserve_range=True)

def norm01(x):
    """Normalizes values in x to be between 0 and 255"""
    r = (x - np.min(x))
    m = np.max(r)
    if m > 0:
        r = np.divide(r, np.max(r))
    return r

def gram_matrix(input):
    # taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    grams = []

    for i in range(a):
        features = input[i].view(b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        grams.append(G.div(b * c * d))

    return grams

def gram_matrix_3d(input):
    # taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    a, b, c, d, e = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    grams = []

    for i in range(a):
        features = input[i].view(b, c * d * e)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        grams.append(G.div(b * c * d * e))

    return grams


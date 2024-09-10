import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.loggers import TensorBoardLogger
from active_dynamicmemory.CardiacActiveDynamicMemory import CardiacActiveDynamicMemory
import pandas as pd
import pytorch_lightning.loggers as pllogging
from . import utils
import numpy as np
import random
import yaml
import torch
import os

N_EPOCHS = 10

BASE_RBACA_EVAL = 'B' # 'R' rbaca or 'B' base or 'E' eval
MODEL_NAME = 'vit' # options {'ResNet50', 'vit'}
    
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

def trained_model(mparams, settings, training=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    settings = argparse.Namespace(**settings)
    os.makedirs(settings.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(settings.TRAINED_MEMORY_DIR, exist_ok=True)
    os.makedirs(settings.RESULT_DIR, exist_ok=True)

    if mparams['task'] == 'cardiac':
        model = CardiacActiveDynamicMemory(mparams=mparams, modeldir=settings.TRAINED_MODELS_DIR, device=device, training=training)
    else:
        raise NotImplementedError('task not implemented')

    # print('runutils1')
    exp_name = get_expname(mparams)
    # print(exp_name)
    weights_path = cached_path(mparams, settings.TRAINED_MODELS_DIR)
    # print(weights_path)

    # print('model', model)
    
    if not os.path.exists(weights_path) and training:
        logger = TensorBoardLogger(settings.LOGGING_DIR, name=exp_name)
        trainer = Trainer(
            max_epochs=N_EPOCHS,
            accelerator='gpu' if torch.cuda.is_available() else None,
            logger=logger,
            val_check_interval=model.mparams.val_check_interval,
            gradient_clip_val=model.mparams.gradient_clip_val,
            strategy='ddp_find_unused_parameters_true'  # Add this line to enable detection of unused parameters in DDP
        )
        trainer.fit(model)
        model.freeze()
        torch.save(model.state_dict(), weights_path)
        if model.mparams.continuous:
            print('train counter', model.train_counter)
            print('label counter', model.trainingsmemory.labeling_counter)
            with open(settings.TRAINED_MEMORY_DIR + exp_name + '.txt', 'w') as f:
                f.write('train counter: ' + str(model.train_counter) + '\n')
                f.write('label counter: ' + str(model.trainingsmemory.labeling_counter))
        if model.mparams.continuous and model.mparams.use_memory:
            save_memory_to_csv(model.trainingsmemory.memorylist, settings.TRAINED_MEMORY_DIR + exp_name + '.csv')
    elif os.path.exists(weights_path):
        print('Read: ' + weights_path)
        state_dict = torch.load(weights_path)
        new_state_dict = dict()
        for k in state_dict.keys():
            if k.startswith('model.'):
                new_state_dict[k.replace("model.", "", 1)] = state_dict[k]
        
        if BASE_RBACA_EVAL != 'E':   
            model.model.load_state_dict(new_state_dict)
        else:
            modified_state_dict = new_state_dict.copy()

            # Get the current model's fully connected layer size
            num_classes_current_model = 9  # The new size for fc.weight and fc.bias

            if MODEL_NAME == 'ResNet50':
                fl_weight = 'fc.weight'
                fl_bias = 'fc.bias'
            elif MODEL_NAME == 'vit':
                fl_weight = 'classifier.weight'
                fl_bias = 'classifier.bias'
                    
            # Extract the weights and biases from new_state_dict
            fc_weight_old = new_state_dict[fl_weight]
            fc_bias_old = new_state_dict[fl_bias]

            # Create new weights and biases with the desired shape and fill with zeros
            fc_weight_new = torch.zeros((num_classes_current_model, fc_weight_old.size(1)))
            fc_bias_new = torch.zeros((num_classes_current_model,))

            # Copy the old weights and biases into the new tensors
            fc_weight_new[:fc_weight_old.size(0), :] = fc_weight_old
            fc_bias_new[:fc_bias_old.size(0)] = fc_bias_old

            # Update the state_dict with the new weights and biases
            modified_state_dict[fl_weight] = fc_weight_new
            modified_state_dict[fl_bias] = fc_bias_new

            # Load the modified state_dict into the model
            model.model.load_state_dict(modified_state_dict)
            
        model.freeze()
    else:
        print(weights_path, 'does not exist')
        model = None
        return model, None, None, exp_name + '.pt'

    print(model.mparams.continuous, model.mparams.use_memory)
    if model.mparams.continuous and model.mparams.use_memory:
        if os.path.exists(settings.TRAINED_MEMORY_DIR + exp_name + '.csv'):
            df_memory = pd.read_csv(settings.TRAINED_MEMORY_DIR + exp_name + '.csv')
        else:
            df_memory = None
            print(settings.TRAINED_MEMORY_DIR + exp_name + '.csv', 'does not exist')
    else:
        df_memory = None

    # always get the last version
    try:
        max_version = max([int(x.split('_')[1]) for x in os.listdir(settings.LOGGING_DIR + exp_name)])
        logs = pd.read_csv(settings.LOGGING_DIR + exp_name + '/version_{}/metrics.csv'.format(max_version))
    except Exception as e:
        print(e)
        logs = None

    return model, logs, df_memory, exp_name + '.pt'

def is_cached(mparams, trained_dir):
    exp_name = get_expname(mparams)
    return os.path.exists(trained_dir + exp_name + '.pt')

def cached_path(mparams, trained_dir):
    exp_name = get_expname(mparams)
    return trained_dir + exp_name + '.pt'

def get_expname(mparams):
    if type(mparams) is argparse.Namespace:
        mparams = vars(mparams).copy()
    elif type(mparams) is AttributeDict:
        mparams = dict(mparams)

    hashed_params = utils.hash(mparams, length=10)

    expname = mparams['task']
    expname += '_cont' if mparams['continuous'] else '_batch'

    if 'naive_continuous' in mparams:
        expname += '_naive'

    expname += '_' + os.path.splitext(os.path.basename(mparams['datasetfile']))[0]
    if mparams['base_model']:
        expname += '_basemodel_' + mparams['base_model'].split('_')[1]
    if mparams['continuous']:
        expname += '_memory' if mparams['use_memory'] else '_nomemory'
        expname += '_tf{}'.format(str(mparams['transition_phase_after']).replace('.', ''))
    else:
        expname += '_' + '-'.join(mparams['noncontinuous_train_splits'])
    expname += '_'+str(mparams['run_postfix'])
    expname += '_'+hashed_params
    return expname

def save_memory_to_csv(memory, savepath):
    if type(memory[0].target) is dict:
        target = [e.target for e in memory]
    else:
        target = [e.target.cpu().numpy() for e in memory]
    df_memory = pd.DataFrame({'filepath':[e.filepath for e in memory],
                             'target': target,
                             'scanner': [e.scanner for e in memory],
                             'pseudodomain': [e.pseudo_domain for e in memory]})
    df_memory.to_csv(savepath, index=False, index_label=False)

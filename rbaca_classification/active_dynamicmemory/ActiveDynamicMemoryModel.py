import argparse
import os
import numpy as np
import pytorch_lightning as pl
import torch
import cv2
from . import utils
# import torch.nn as nn
from active_dynamicmemory.ActiveDynamicMemory import RBACADynamicMemory
from active_dynamicmemory.ActiveDynamicMemory import MemoryItem
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import random
import yaml
import torchvision.transforms.functional as TF

MODEL_NAME = 'ResNet50' # options {'ResNet50', 'vit'}
BASE_RBACA_EVAL = 'B' # 'R' rbaca or 'B' base or 'E' eval
DATA_AUG = False



if BASE_RBACA_EVAL == 'E':
    DATA_AUG = False

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

if MODEL_NAME == 'vit':
    from transformers import ViTImageProcessor
    # from transformers import ViTFeatureExtractor
    ViT_IMAGE_DIM = 224
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

def inverse_norm01(r, min_x, max_x):
    # r is the normalized data (between 0 and 1)
    # min_x and max_x are the original minimum and maximum values of x
    original_range = max_x - min_x
    return r * original_range + min_x

def process_batch(batch):
    batch = [inverse_norm01(image, 0, 1) for image in batch]
    
    batch = torch.stack(batch)
    batch = batch.unsqueeze(1)
    batch = torch.cat([batch, batch, batch], dim=1)

    # Process images using the ViT processor
    # processed_inputs = feature_extractor(images=batch, return_tensors="pt", do_normalize=True, do_rescale=False)
    processed_inputs = processor(images=batch, return_tensors="pt", do_normalize=True, do_rescale=False)

    batch = processed_inputs['pixel_values']
        
    return batch

def custom_transform(images, batch_size=64):
    # if DATA_AUG:
    #     batch_size = 1
            
    # Create a DataLoader for batching
    dataset = TensorDataset(torch.tensor(images))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # print(f'batch_size custom_transformBD = {batch_size}')
    
    all_processed_images = []
    
    for batch in dataloader:
        batch = batch[0]  # Extract the batch from the tuple
        # batch = batch.to(device)
        # print(f'Real batch_size custom_transformBD = {batch.shape[0]}')
        processed_batch = process_batch(batch)
        all_processed_images.append(processed_batch)
    
    # Concatenate all processed batches
    all_processed_images = torch.cat(all_processed_images, dim=0)
    
    return all_processed_images



def data_augmentation_random(batch_images):
    # Ensure batch_images is on the GPU
    if not batch_images.is_cuda:
        batch_images = batch_images.cuda()
    
    # pltt = True
    
    # MODEL_NAME can be either 'vit' or 'ResNet50'
    batch_size, channels, height, width = batch_images.shape

    # Initialize a list to store augmented images
    augmented_images = []    

    for image in batch_images:
        
        random_int = random.randint(0, 9)
        
        image = image * 255  # Scale image to [0, 255] range
        
        if random_int == 0:
            image_new = image
        elif random_int == 1:
            image_new = torch.rot90(image, 2, [1, 2])  # Rotate 180 degrees
        elif random_int == 2:
            image_new = torch.flip(image, [2])  # Horizontal flip
        elif random_int == 3:   
            image_new = torch.flip(image, [1])  # Vertical flip
        elif random_int == 4:   
            # Blurring using Gaussian kernel
            image_new = TF.gaussian_blur(image.unsqueeze(0), kernel_size=15, sigma=1.5).squeeze(0)
        elif random_int == 5:
            # Brightness change (increase by 10%)
            image_new = torch.clamp(image + 25.5, 0, 255)
        elif random_int == 6:   
            # Brightness change (decrease by 10%)
            image_new = torch.clamp(image - 25.5, 0, 255)
        elif random_int == 7:   
            # Contrast adjustment (increase by 10%)
            mean = image.mean()
            image_new = torch.clamp((image - mean) * 1.1 + mean, 0, 255)
        elif random_int == 8: 
            # Contrast adjustment (decrease by 10%)
            mean = image.mean()
            image_new = torch.clamp((image - mean) * 0.9 + mean, 0, 255)
        else: 
            # Gaussian noise addition
            noise = torch.randn_like(image) * (255 * 0.035)
            image_new = torch.clamp(image + noise, 0, 255)

        image_new = image_new / 255.0  # Scale image back to [0, 1] range
        if MODEL_NAME == 'vit':
            image_new = torch.mean(image_new, dim=0)
            image_new = (image_new + 1) / 2 # -1,1 to 0,1
            
        augmented_images.append(image_new)

    # Convert list of augmented images to tensor
    augmented_images = torch.stack(augmented_images)
    if MODEL_NAME == 'vit':
        augmented_images = torch.clamp(augmented_images, min=0.0, max=1.0)
        augmented_images = custom_transform(augmented_images)

    data_aug_factor = 1
        
    augmented_images = augmented_images.reshape(batch_size * data_aug_factor, channels, height, width)
    
    return augmented_images

class ActiveDynamicMemoryModel(pl.LightningModule, ABC):

    def init(self, mparams={}, modeldir=None, device=torch.device('cpu'), training=True):
        self.mparams = argparse.Namespace(**mparams)
        self.to(device)

        self.modeldir = modeldir
        
        self.mparams.seed = 42  # Replace 42 with your desired seed value

        self.model, self.stylemodel, self.gramlayers = self.load_model_stylemodel(self.mparams.droprate, load_stylemodel=training)
        if training:
            self.stylemodel.to(device)

        self.learning_rate = self.mparams.learning_rate
        self.train_counter = 0

        if self.mparams.continuous:
            self.budget = self.mparams.startbudget
            self.budgetchangecounter = 0
            if self.mparams.allowedlabelratio == 0:
                self.budgetrate = 0
            else:
                self.budgetrate = 1 / self.mparams.allowedlabelratio


        if not self.mparams.base_model is None and training:
            state_dict =  torch.load(os.path.join(modeldir, self.mparams.base_model))
            new_state_dict = {}
            for key in state_dict.keys():
                if key.startswith('model.'):
                    new_state_dict[key.replace('model.', '', 1)] = state_dict[key]
                    
            if BASE_RBACA_EVAL != 'E':   
                self.model.load_state_dict(new_state_dict)
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
                self.model.load_state_dict(modified_state_dict)

        # Initilize checkpoints to calculate BWT, FWT after training
        self.scanner_checkpoints = dict()
        self.scanner_checkpoints[self.mparams.order[0]] = True
        for scanner in self.mparams.order[1:]:
            self.scanner_checkpoints[scanner] = False

        if self.mparams.use_memory and self.mparams.continuous and training:
            self.init_memory_and_gramhooks()
                        

    def init_memory_and_gramhooks(self):
        self.grammatrices = []

        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

        initelements = self.getmemoryitems_from_base(num_items=self.mparams.memorymaximum)

        if self.mparams.method == 'RBACA':
            self.trainingsmemory = RBACADynamicMemory(initelements=initelements, 
                                                      memorymaximum=self.mparams.memorymaximum,
                                                      seed=self.mparams.seed,
                                                      perf_queue_len=self.mparams.len_perf_queue,
                                                     transformgrams=self.mparams.transformgrams,
                                                     outlier_distance=self.mparams.outlier_distance)
            self.insert_element = self.insert_element_rbaca

    def getmemoryitems_from_base(self, num_items=128):
        batch_size = self.mparams.batch_size
        # if DATA_AUG:
        #     batch_size = 1
        
        dl = DataLoader(self.TaskDatasetBatch(self.mparams.datasetfile,
                                            iterations=None,
                                            batch_size=batch_size,
                                            split=['base']),
                            batch_size=batch_size, num_workers=4, pin_memory=True, collate_fn=self.collate_fn)
        
        # print(f'batch_size getmemoryitems_from_base = {batch_size}')
        
        memoryitems = []
        self.grammatrices = []

        self.stylemodel.eval()

        for batch in dl:
            torch.cuda.empty_cache()

            x, y, scanner, filepath = batch
            
            # if DATA_AUG:
            #     x = data_augmentation_random(x)
            
            # print(f'Real batch_size getmemoryitems_from_base = {x.shape[0]}')
            # print('getmemoryitems_from_base: x.shape:', x.shape, ', y.shape:', y.shape)

            if type(x) is list or type(x) is tuple:
                xstyle = torch.stack(x)
            elif x.size()[1]==1 and self.mparams.dim!=3:
                xstyle = torch.cat([x, x, x], dim=1)
            else:
                xstyle = x


            _ = self.stylemodel(xstyle.to(self.device))

            for i, f in enumerate(filepath):
                target = y[i]
                if type(target) == torch.Tensor:
                    det_target = target.detach().cpu()
                else:
                    det_target = {}
                    for k, v in target.items():
                        det_target[k] = v.detach().cpu()
                grammatrix = self.grammatrices[0][i].detach().cpu().numpy().flatten()
                # print('getmemoryitems_from_base: det_target:', det_target)
                memoryitems.append(MemoryItem(x[i].detach().cpu(), det_target, f, scanner[i],
                                              current_grammatrix=grammatrix,
                                              pseudo_domain=0))

            if len(memoryitems) >= num_items:
                break

            self.grammatrices = []

        return memoryitems[:num_items]

    def gram_hook(self,  m, input, output):
        if self.mparams.dim == 2:
            self.grammatrices.append(utils.gram_matrix(input[0]))
        elif self.mparams.dim == 3:
            self.grammatrices.append(utils.gram_matrix_3d(input[0]))
        else:
            raise NotImplementedError(f'gram hook with {self.mparams.dim} dimensions not defined')

    # This is called when mparams.method == 'RBACA'
    def insert_element_rbaca(self, x, y, filepath, scanner):
        budget_before = self.budget

        for i, img in enumerate(x):
            grammatrix = self.grammatrices[0][i].detach().cpu().numpy().flatten()

            target = y[i]
            if type(target) == torch.Tensor:
                det_target = target.detach().cpu()
            else:
                det_target = {}
                for k, v in target.items():
                    det_target[k] = v.detach().cpu()

            # print('insert_element_rbaca: det_target:', det_target)
            new_mi = MemoryItem(img.detach().cpu(), det_target, filepath[i], scanner[i], grammatrix)
            self.budget = self.trainingsmemory.insert_element(new_mi, self.budget, self)

        self.budget = self.trainingsmemory.check_outlier_memory(self.budget, self)
        self.trainingsmemory.counter_outlier_memory()

        if budget_before == self.budget:
            self.budgetchangecounter += 1
        else:
            self.budgetchangecounter = 1

        # form trainings X domain balanced batches to train one epoch on all newly inserted samples
        if not np.all(list(
                self.trainingsmemory.domaincomplete.values())) and self.budgetchangecounter < 10:  # only train when a domain is incomplete and new samples are inserted?

            for k, v in self.trainingsmemory.domaincomplete.items():
                # print(k)
                if not v:
                    # print('self.trainingsmemory.domainMetric[' + str(k) + '] = ' + str(self.trainingsmemory.domainMetric[k]))
                    if len(self.trainingsmemory.domainMetric[k]) == self.mparams.len_perf_queue:
                        mean_metric = np.mean(self.trainingsmemory.domainMetric[k])
                        # print(k, mean_metric)
                        if self.completed_domain(mean_metric):
                            # print('entrei if')
                            self.trainingsmemory.domaincomplete[k] = True
                            
            # print('sai loop')
            return True
        else:
            return False

   

    def training_step(self, batch, batch_idx):
        x, y, scanner, filepath = batch
        # print(f'Real batch_size training_step = {x.shape[0]}')
        
        if DATA_AUG:
            x = data_augmentation_random(x).to(self.device)
        

        self.grammatrices = []

        self.stylemodel.eval()

        if self.mparams.continuous:
            # save checkpoint at scanner shift
            newshift = False
            for s in scanner:
                if s != self.mparams.order[0] and not self.scanner_checkpoints[s]:
                    newshift = True
                    shift_scanner = s
            if newshift:
                # print('new shift to', shift_scanner)
                exp_name = utils.get_expname(self.mparams)
                weights_path = self.modeldir + exp_name + '_shift_' + shift_scanner + '.pt'
                torch.save(self.model.state_dict(), weights_path)
                self.scanner_checkpoints[shift_scanner] = True
        
        # print('training_step1')

        if self.mparams.use_memory:
            #y = y[:, None]
            self.grammatrices = []
            if type(x) is list or type(x) is tuple:
                xstyle = torch.stack(x)
            elif x.size()[1] == 1 and self.mparams.dim != 3:
                xstyle = torch.cat([x, x, x], dim=1)
            else:
                xstyle = x
            _ = self.stylemodel(xstyle)

            train_a_step = self.insert_element(x, y, filepath, scanner)
            # print('train_a_step')
            # print(train_a_step)

            if train_a_step:
                batch_size = self.mparams.batch_size
                # if DATA_AUG:
                #     batch_size = 1
                x, y = self.trainingsmemory.get_training_batch(batch_size,
                                                                 batches=int(
                                                                     self.mparams.training_batch_size / batch_size))
                self.train_counter += 1
                
                # print('x')
                # # print(x)
                # print('y')
                # # print(y)
                # print('self.train_counter')
                # print(self.train_counter)
                
            else:
                return None

        loss = self.get_task_loss(x, y)
        
        self.log('train_loss', loss)
        self.grammatrices = []
        # print('loss')
        # print(loss)
        return loss
        # return None

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        # x = x.unsqueeze(1)
        if MODEL_NAME == 'ResNet50':
            x = torch.cat([x, x, x], dim=1)
        
        # print(f'x1.shape = {x.shape}')
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        batch_size = self.mparams.batch_size
        # if DATA_AUG:
        #     batch_size = 1
        # print(f'batch_size train_dataloader = {batch_size}')
            
        if self.mparams.continuous:
            if 'randomstream' in self.mparams:
                return DataLoader(self.TaskDatasetContinuous(self.mparams.datasetfile,
                                                             transition_phase_after=self.mparams.transition_phase_after,
                                                             seed=self.mparams.seed,
                                                             order=self.mparams.order, random=True, model=self.model),
                                  batch_size=batch_size, num_workers=8, drop_last=True,
                                  collate_fn=self.collate_fn)
            else:
                return DataLoader(self.TaskDatasetContinuous(self.mparams.datasetfile,
                                                             transition_phase_after=self.mparams.transition_phase_after,
                                                             seed=self.mparams.seed,
                                                             order=self.mparams.order, model=self.model),
                                  batch_size=batch_size, num_workers=8, drop_last=True,
                                  collate_fn=self.collate_fn)
        else:
            return DataLoader(self.TaskDatasetBatch(self.mparams.datasetfile,
                                                    iterations=self.mparams.noncontinuous_steps,
                                                    batch_size=batch_size,
                                                    split=self.mparams.noncontinuous_train_splits,
                                                    res=self.mparams.scanner,
                                                    seed=self.mparams.seed),
                              batch_size=batch_size, num_workers=8, collate_fn=self.collate_fn)

    #@pl.data_loader
    def val_dataloader(self):
        batch_size = 4
        # if DATA_AUG:
        #     batch_size = 1
            
        # print(f'batch_size val_dataloader = {batch_size}')
            
        return DataLoader(self.TaskDatasetBatch(self.mparams.datasetfile,
                                                split='val', res=self.mparams.order),
                          batch_size=batch_size,
                          num_workers=2,
                          collate_fn=self.collate_fn)




    @abstractmethod
    def load_model_stylemodel(self, droprate, load_stylemodel=False):
        pass

    @abstractmethod
    def get_task_metric(self, image, target):
        pass

    @abstractmethod
    def completed_domain(self, m):
        pass

    @abstractmethod
    def get_task_loss(self, x, y):
        pass

    @abstractmethod
    def get_uncertainties(self, x):
        pass
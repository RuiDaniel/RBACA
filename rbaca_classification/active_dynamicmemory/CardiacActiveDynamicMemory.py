import torch
import torch.nn as nn
from datasets.ContinuousDataset import CardiacContinuous
from datasets.BatchDataset import CardiacBatch
# from monai.metrics import DiceMetric
# import monai.networks.utils as mutils
from torch.utils.data import DataLoader, TensorDataset
from active_dynamicmemory.ActiveDynamicMemoryModel import ActiveDynamicMemoryModel
# import monai.networks.nets as monaimodels
import torchvision.models as models
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from transformers import ViTForImageClassification
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt

MODEL_NAME = 'vit' # options {'ResNet50', 'vit'}
FINAL_LAYER = 'Static' # options {'Static', 'Dynamic'}
BASE_RBACA_EVAL = 'B' # 'C' rbaca or 'B' base or 'E' eval
DATA_AUG = False

if BASE_RBACA_EVAL == 'E':
    DATA_AUG = False


BASE_MAPPING = [1, 2, 4, 0, -5, 6, -7, 3, 5] # [2,3,0,1] if {0: 2, 1: 3, 2: 0, 3: 1}
RBACA_MAPPING = [1, 2, 4, 0, 8, 6, 7, 3, 5]

N_CLASSES = 9 # NEW

###

classes_observed = set()
classes_observed_mapping = {}
for i in range(N_CLASSES):
    classes_observed_mapping[i] = -i - 1
    


classes = ['HCM', 'DCM', 'HHD', 'ARV', 'AHS', 'IHD', 'LVNC', 'NOR', 'Other']    

# Mapping of labels to numerical values
label_mapping = {
    'HCM': 0,
    'DCM': 1,
    'HHD': 2,
    'ARV': 3,
    'AHS': 4,
    'IHD': 5,
    'LVNC': 6,
    'NOR': 7,
    'Other': 8
}

###



def classes_observed_mapping_fill(new):
    for i in range(N_CLASSES):
        classes_observed_mapping[i] = new[i]
        
        
INIT_SIZE_DYNAMIC = 1

if BASE_RBACA_EVAL == 'C' and FINAL_LAYER == 'Dynamic':
    INIT_SIZE_DYNAMIC = len([x for x in BASE_MAPPING if x >= 0])
    classes_observed = set([i for i, x in enumerate(BASE_MAPPING) if x >= 0])
    classes_observed_mapping_fill(BASE_MAPPING)
elif BASE_RBACA_EVAL == 'E' and FINAL_LAYER == 'Dynamic':
    INIT_SIZE_DYNAMIC = len([x for x in RBACA_MAPPING if x >= 0])
    classes_observed = set([i for i, x in enumerate(RBACA_MAPPING) if x >= 0])
    classes_observed_mapping_fill(RBACA_MAPPING)
    
    
    
print(f'classes_observed = {classes_observed}')
print(f'classes_observed_mapping = {classes_observed_mapping}')

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

def inverse_norm01(r, min_x, max_x):
    # r is the normalized data (between 0 and 1)
    # min_x and max_x are the original minimum and maximum values of x
    original_range = max_x - min_x
    return r * original_range + min_x

if MODEL_NAME == 'vit':
    from transformers import ViTImageProcessor
    # from transformers import ViTFeatureExtractor
    ViT_IMAGE_DIM = 224
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    
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

import torchvision.transforms.functional as TF


def plot_images(image):
    image = image * 255  # Scale image to [0, 255] range
    
    images = []
    
    image_new = image
    images.append(image_new)
    
    image_new = torch.rot90(image, 2, [1, 2])  # Rotate 180 degrees
    images.append(image_new)

    image_new = torch.flip(image, [2])  # Horizontal flip
    images.append(image_new)

    image_new = torch.flip(image, [1])  # Vertical flip
    images.append(image_new)

    # Blurring using Gaussian kernel
    image_new = TF.gaussian_blur(image.unsqueeze(0), kernel_size=15, sigma=1.5).squeeze(0)
    images.append(image_new)

    # Brightness change (increase by 10%)
    image_new = torch.clamp(image + 25.5, 0, 255)
    images.append(image_new)

    # Brightness change (decrease by 10%)
    image_new = torch.clamp(image - 25.5, 0, 255)
    images.append(image_new)

    # Contrast adjustment (increase by 10%)
    mean = image.mean()
    image_new = torch.clamp((image - mean) * 1.1 + mean, 0, 255)
    images.append(image_new)
    
    # Contrast adjustment (decrease by 10%)
    mean = image.mean()
    image_new = torch.clamp((image - mean) * 0.9 + mean, 0, 255)
    images.append(image_new)

    # Gaussian noise addition
    noise = torch.randn_like(image) * (255 * 0.035)
    image_new = torch.clamp(image + noise, 0, 255)
    images.append(image_new)
    
    images = [image.cpu().detach().numpy() for image in images]    
    images = np.array(images)
    
    images = images / 255.0  # Scale image back to [0, 1] range
    
    
    print(f'max = {np.max(images)}, min = {np.min(images)}')
    
    # Titles for the images
    titles = [
        'Original', 'Rotated 180', 'Flipped Horizontal', 'Flipped Vertical',
        'Blurred', 'Brightness Increase', 'Brightness Decrease', 
        'Contrast Increase', 'Contrast Decrease', 'Noisy'
    ]

    # Create a figure to plot the images
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    # Plot each image with its corresponding title
    for i in range(10):
        image = images[i]
        # print(image.shape)
        if MODEL_NAME == 'ResNet50':
            image = np.squeeze(image)
        # print(image.shape)
        ax = axes.flat[i]
        ax.imshow(image, cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')  # Hide axes

    # Hide the last subplot as there are only 10 images
    axes.flat[-1].axis('off')
    axes.flat[-2].axis('off')

    # Save the plot to a .png file
    plt.tight_layout()
    plt.savefig('aug.png')
    # plt.show()
    
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

    
class CardiacActiveDynamicMemory(ActiveDynamicMemoryModel):

    def __init__(self, mparams={}, modeldir=None, device=torch.device('cpu'), training=True):
        super(ActiveDynamicMemoryModel, self).__init__()
        self.collate_fn = None
        self.TaskDatasetBatch = CardiacBatch
        self.TaskDatasetContinuous = CardiacContinuous
        self.loss = nn.CrossEntropyLoss() # NEW
        
        self.init(mparams=mparams, modeldir=modeldir, device=device, training=training)
        # print('Fim de CADM')
        

    def load_model_stylemodel(self, droprate, load_stylemodel=False):
        
        if FINAL_LAYER == 'Static':
            initial_size_final_layer = N_CLASSES
        elif FINAL_LAYER == 'Dynamic':
            initial_size_final_layer = INIT_SIZE_DYNAMIC
        else:
            print('ERROR, Invalid Mode')
            exit(0)
        
        print(f'initial_size_final_layer = {initial_size_final_layer}')
        if MODEL_NAME == 'ResNet50':
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, initial_size_final_layer)
            
        elif MODEL_NAME == 'vit':
            dim = 768
            # model = CustomViTForImageClassification(image_size=(ViT_IMAGE_DIM, ViT_IMAGE_DIM), num_classes=initial_size_final_layer, pretrained_model_name='google/vit-base-patch16-224')
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            model.classifier = nn.Linear(dim, initial_size_final_layer)
        else:
            print('ERROR, Invalid Model')
            exit(0)
            
        # print('Model got .fc')
        
        if load_stylemodel:
            stylemodel = models.resnet50(pretrained=True)
            gramlayers = [stylemodel.layer2[-1].conv1]
            stylemodel.eval()

            return model, stylemodel, gramlayers

        return model, None, None

    

    def evaluate_f1_score(self, test_labels, predicted_labels): # NEW
        # Compute F1 score
        # mean_f1_score = f1_score(test_labels, predicted_labels, average='macro')
        
        if isinstance(test_labels, torch.Tensor):
            test_labels = test_labels.cpu().numpy()
        if isinstance(predicted_labels, torch.Tensor):
            predicted_labels = predicted_labels.cpu().numpy()
        
        # Compute F1 score for each class
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            f1_score_classes_aux = f1_score(test_labels, predicted_labels, average=None, labels = [0,1,2,3,4,5,6,7,8])
            
        
        # sum_f1 = 0.0
        # count = 0.0
        for i in range(len(f1_score_classes_aux)):
            if i not in test_labels:
                f1_score_classes_aux[i] = None
            # else:
                # sum_f1 += f1_score_classes_aux[i]
                # count += 1
        
        if FINAL_LAYER == 'Static':
            f1_score_classes = f1_score_classes_aux
        elif FINAL_LAYER == 'Dynamic':
            # f1_score_classes_aux = [0,1,2,3,4,5,6,7,8]
            # classes_observed_mapping[7]=2
            f1_score_classes = [0 for _ in range(N_CLASSES)]
            for i in range(N_CLASSES):
                for j in range(N_CLASSES):
                    k = -1
                    if classes_observed_mapping[i] == j:
                        k = j
                        break
                if k != -1:
                    f1_score_classes[i] = f1_score_classes_aux[j]
                else:
                    f1_score_classes[i] = None
            
            for i in range(N_CLASSES):
                if -i - 1 in test_labels:
                    f1_score_classes[i] = 0.0
        else:
            print('ERROR')
            exit(0)
        
        f1_score_classes = np.array(f1_score_classes)   
        
        f1_score_classes = np.array([np.nan if x is None else x for x in f1_score_classes])
        
        mean_f1_score = np.nanmean(f1_score_classes)
        
        
        return mean_f1_score, f1_score_classes

    def get_task_metric(self, image, target):
        """
        Task metric for cardiac classification is the f1 score
        """
        self.eval()
        x = image
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        if MODEL_NAME == 'ResNet50':
            x = x.unsqueeze(1)
            x = torch.cat([x, x, x], dim=1)
        elif MODEL_NAME == 'vit':
            x = x.unsqueeze(0)
            
        y = torch.tensor([target]).to(self.device)
        # print('task_metric: x.shape:', x.shape, ', y.shape:', y.shape)
        
        if FINAL_LAYER == 'Dynamic':
            y = [classes_observed_mapping[t.item()] for t in y]
            y = torch.tensor(y, dtype=torch.long).to(self.device)
        
        # print(f'x2.shape = {x.shape}')
        y_hat = self.model(x.float())
        if MODEL_NAME == 'vit':
            y_hat = y_hat.logits
            
        y_hat_flat = F.softmax(y_hat, dim=1)
        _, y_hat_flat = y_hat.max(dim=1)
        
       
        
        # print('task_metric pre f1: y:', y, ', y_hat_flat:', y_hat_flat, 'y.shape:', y.shape, ', y_hat_flat.shape:', y_hat_flat.shape)

        # Assuming one_hot_pred and one_hot_target are already on GPU
        f1, _ = self.evaluate_f1_score(y, y_hat_flat)
        
        self.train()
        #return dice.detach().cpu().numpy()
        return f1

    def completed_domain(self, m):
        """
        Domain is completed if m larger than a threshold
        :param m: value to compare to the threshold
        :return: Wheter or not the domain is considered completed
        """
        if np.isnan(m):
            # print('apanhei o nan1')
            return False
        
        return m>self.mparams.completion_limit

    def validation_step(self, batch, batch_idx):
        """
        Validation step managed by pytorch lightning
        :param batch:
        :param batch_idx:
        :return: logs for validation
        """
        global classes_observed
        
        # print('VALIDATION_STEP')
        x, y, res, img = batch
        
        if DATA_AUG:
            x = data_augmentation_random(x).to(self.device)
        
        # print(f'Real batch_size validation_step = {x.shape[0]}')
        
        # if DATA_AUG:
        #     if len(x.shape) == 4:
        #         x = x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])
        #         # print(f'x.shape = {x.shape}')
            
        #     if len(y.shape) == 2:
        #         y = y.view(y.shape[0] * y.shape[1], 1)
        #         # print(f'y.shape = {y.shape}')
        
        self.grammatrices = []
        
        if FINAL_LAYER == 'Dynamic': # NEW
            label_tensor = y  # Convert label to a tensor
            centre_classes_observed = set(torch.unique(label_tensor).tolist())
            
            # print('Before: classes_observed', classes_observed, ', len(classes_observed)', len(classes_observed))
            # print(f'centre_classes_observed = {centre_classes_observed}')
            
            new_classes = centre_classes_observed - classes_observed
            
            centres_observed_old = classes_observed
            
            classes_observed = classes_observed | centre_classes_observed
            
            if len(new_classes) > 0:
                print('After: new_classes', new_classes) 
                print('After: classes_observed', classes_observed)
                
                for i, elem in enumerate(new_classes):
                    print(f'classes_observed_mapping[{elem}] = {i + len(centres_observed_old)}')
                    classes_observed_mapping[elem] = i + len(centres_observed_old)
                    
                    
                if self.model == None:
                    print('No model yet!!!!!!!!!!')
                elif MODEL_NAME == 'ResNet50':
                    # Update the model's final layer
                    # print(f'Before, weights: {self.model.fc.weight}, bias: {self.model.fc.bias}')
                
                    old_weights = self.model.fc.weight.data.clone()
                    old_biases = self.model.fc.bias.data.clone()
                    new_fc = nn.Linear(self.model.fc.in_features, len(classes_observed)).to(self.device)
                    
                    if old_biases.size(0) > 0:
                        # Copy the weights and biases from the old layer to the new layer
                        print(f'old_weights.shape = {old_weights.shape}, new_fc.weight.shape = {new_fc.weight.shape}')
                        with torch.no_grad():
                            new_fc.weight[:old_weights.size(0)] = old_weights
                            new_fc.bias[:old_biases.size(0)] = old_biases

                    # Replace the old fully connected layer with the new layer
                    self.model.fc = new_fc
                    
                    # print(f'After, weights: {self.model.fc.weight}, bias: {self.model.fc.bias}')
                    print('After: classes_observed_mapping', classes_observed_mapping)
                elif MODEL_NAME == 'vit':
                    # Update the model's final layer
                    # print(f'Before, weights: {self.model.classifier.weight}, bias: {self.model.classifier.bias}')
                
                    old_weights = self.model.classifier.weight.data.clone()
                    old_biases = self.model.classifier.bias.data.clone()
                    new_classifier = nn.Linear(self.model.classifier.in_features, len(classes_observed)).to(self.device)
                    
                    if old_biases.size(0) > 0:
                        # Copy the weights and biases from the old layer to the new layer
                        print(f'old_weights.shape = {old_weights.shape}, new_classifier.weight.shape = {new_classifier.weight.shape}')
                        with torch.no_grad():
                            new_classifier.weight[:old_weights.size(0)] = old_weights
                            new_classifier.bias[:old_biases.size(0)] = old_biases

                    # Replace the old fully connected layer with the new layer
                    self.model.classifier = new_classifier
                    
                    # print(f'After, weights: {self.model.classifier.weight}, bias: {self.model.classifier.bias}')
                    print('After: classes_observed_mapping', classes_observed_mapping)
                print()
            

        y_hat = self.forward(x.float())
        if MODEL_NAME == 'vit':
            y_hat = y_hat.logits
        
        res = res[0]
        
        # y_toloss = y.unsqueeze(1)
        
        # print('validation_step: x.shape:', x.shape, ', y.shape:', y.shape)
        # print('x.shape', x.shape)
        # print('y.shape', y.shape)
        
        # print('ytrue.shape: ', y.shape, ', ypred.shape: ', y_hat.shape)
        
        # loss = self.loss(y_hat, y)
        
        if FINAL_LAYER == 'Dynamic':
            mapped_target = [classes_observed_mapping[t.item()] for t in y]
            mapped_target = torch.tensor(mapped_target, dtype=torch.long).to(self.device)
            # print('ytrue_mapped: ', mapped_target, ', ypred: ', y_hat)
            loss = self.loss(y_hat, mapped_target)
        elif FINAL_LAYER == 'Static':
            # if MODEL_NAME == 'ResNet50':
            y = y.squeeze(dim=1)
            # print('ytrue.shape: ', y.shape, ', ypred.shape: ', y_hat.shape)
            # print('ytrue[0]: ', y[0], ', ypred[0]: ', y_hat[0])
            loss = self.loss(y_hat, y)
        else:
            print('ERROR')
            exit(0)
                
        # print('loss1', loss)
        
        y_hat_flat = F.softmax(y_hat, dim=1)
        _, y_hat_flat = y_hat.max(dim=1)
        
        # print('y.shape', y.shape)
        # print('y_hat_flat.shape', y_hat_flat.shape)
        
        f1, f1_classes = self.evaluate_f1_score(y, y_hat_flat)
        
        # print('f1', f1)
        # print('f1_classes', f1_classes)
        
        # input()

        # classes = ['HCM', 'DCM', 'HHD', 'ARV', 'AHS', 'IHD', 'LVNC', 'NOR', 'Other']
        self.log_dict({f'val_loss_{res}': loss,
                       f'val_f1_HCM_{res}': f1_classes[0],
                       f'val_f1_DCM_{res}': f1_classes[1],
                       f'val_f1_HHD_{res}': f1_classes[2],
                       f'val_f1_ARV_{res}': f1_classes[3],
                       f'val_f1_AHS_{res}': f1_classes[4],
                       f'val_f1_IHD_{res}': f1_classes[5],
                       f'val_f1_LVNC_{res}': f1_classes[6],
                       f'val_f1_NOR_{res}': f1_classes[7],
                       f'val_f1_Other_{res}': f1_classes[8]})


    def get_task_loss(self, xs, ys):
        global classes_observed
        
        if FINAL_LAYER == 'Dynamic': # NEW
            label_tensor = ys  # Convert label to a tensor
            # print(f'label_tensor = {label_tensor}')
            label_tensor = torch.tensor(label_tensor)
            centre_classes_observed = set(torch.unique(label_tensor).tolist())
            
            # print('Before: classes_observed', classes_observed, ', len(classes_observed)', len(classes_observed))
            # print(f'centre_classes_observed = {centre_classes_observed}')
            
            new_classes = centre_classes_observed - classes_observed
            
            centres_observed_old = classes_observed
            
            classes_observed = classes_observed | centre_classes_observed
            
            if len(new_classes) > 0:
                print('After: new_classes', new_classes) 
                print('After: classes_observed', classes_observed)
                
                for i, elem in enumerate(new_classes):
                    print(f'classes_observed_mapping[{elem}] = {i + len(centres_observed_old)}')
                    classes_observed_mapping[elem] = i + len(centres_observed_old)
                    
                    
                if self.model == None:
                    print('No model yet!!!!!!!!!!')
                elif MODEL_NAME == 'ResNet50':
                    # Update the model's final layer
                    # print(f'Before, weights: {self.model.fc.weight}, bias: {self.model.fc.bias}')
                
                    old_weights = self.model.fc.weight.data.clone()
                    old_biases = self.model.fc.bias.data.clone()
                    new_fc = nn.Linear(self.model.fc.in_features, len(classes_observed)).to(self.device)
                    
                    if old_biases.size(0) > 0:
                        # Copy the weights and biases from the old layer to the new layer
                        print(f'old_weights.shape = {old_weights.shape}, new_fc.weight.shape = {new_fc.weight.shape}')
                        with torch.no_grad():
                            new_fc.weight[:old_weights.size(0)] = old_weights
                            new_fc.bias[:old_biases.size(0)] = old_biases

                    # Replace the old fully connected layer with the new layer
                    self.model.fc = new_fc
                    
                    # print(f'After, weights: {self.model.fc.weight}, bias: {self.model.fc.bias}')
                    print('After: classes_observed_mapping', classes_observed_mapping)
                elif MODEL_NAME == 'vit':
                    # Update the model's final layer
                    # print(f'Before, weights: {self.model.classifier.weight}, bias: {self.model.classifier.bias}')
                
                    old_weights = self.model.classifier.weight.data.clone()
                    old_biases = self.model.classifier.bias.data.clone()
                    new_classifier = nn.Linear(self.model.classifier.in_features, len(classes_observed)).to(self.device)
                    
                    if old_biases.size(0) > 0:
                        # Copy the weights and biases from the old layer to the new layer
                        print(f'old_weights.shape = {old_weights.shape}, new_classifier.weight.shape = {new_classifier.weight.shape}')
                        with torch.no_grad():
                            new_classifier.weight[:old_weights.size(0)] = old_weights
                            new_classifier.bias[:old_biases.size(0)] = old_biases

                    # Replace the old fully connected layer with the new layer
                    self.model.classifier = new_classifier
                    
                    # print(f'After, weights: {self.model.classifier.weight}, bias: {self.model.classifier.bias}')
                    print('After: classes_observed_mapping', classes_observed_mapping)
                print()
                        
        if type(xs) is list:
            loss = None
            for i, x in enumerate(xs):
                y = ys[i]

                y = torch.stack(y).to(self.device)
                x = x.to(self.device)
                y_hat = self.forward(x.float())
                if MODEL_NAME == 'vit':
                    y_hat = y_hat.logits
                
                
                
                if loss is None:
                    # y_toloss = y.unsqueeze(1)
                    # loss = self.loss(y_hat, y)
                    if FINAL_LAYER == 'Dynamic':
                        mapped_target = [classes_observed_mapping[t.item()] for t in y]
                        mapped_target = torch.tensor(mapped_target, dtype=torch.long).to(self.device)
                        loss = self.loss(y_hat, mapped_target)
                    elif FINAL_LAYER == 'Static':
                        # if MODEL_NAME == 'ResNet50':
                        y = y.squeeze(dim=1)
                        loss = self.loss(y_hat, y)
                    else:
                        print('ERROR')
                        exit(0)
                    # print('loss2', loss)
                else:
                    # y_toloss = y.unsqueeze(1)
                    # loss += self.loss(y_hat, y)
                    # print('loss3', loss)
                    if FINAL_LAYER == 'Dynamic':
                        mapped_target = [classes_observed_mapping[t.item()] for t in y]
                        mapped_target = torch.tensor(mapped_target, dtype=torch.long).to(self.device)
                        loss += self.loss(y_hat, mapped_target)
                    elif FINAL_LAYER == 'Static':
                        # if MODEL_NAME == 'ResNet50':
                        y = y.squeeze(dim=1)
                        loss += self.loss(y_hat, y)
                    else:
                        print('ERROR')
                        exit(0)
        else:
            if type(ys) is list:
                ys = torch.stack(ys).to(self.device)
            y_hat = self.forward(xs.float())
            if MODEL_NAME == 'vit':
                y_hat = y_hat.logits
            
            # y_toloss = ys.unsqueeze(1)
            # loss = self.loss(y_hat, ys)
            # print('loss4', loss)
            if FINAL_LAYER == 'Dynamic':
                mapped_target = [classes_observed_mapping[t.item()] for t in ys]
                mapped_target = torch.tensor(mapped_target, dtype=torch.long).to(self.device)
                loss = self.loss(y_hat, mapped_target)
            elif FINAL_LAYER == 'Static':
                ys = ys.squeeze(dim=1)
                loss = self.loss(y_hat, ys)
            else:
                print('ERROR')
                exit(0)
        
        # print('loss', loss)

        return loss

    def get_uncertainties(self, x):
        y_hat_flats = []
        uncertainties = []

        for i in range(self.mparams.uncertainty_iterations):
            outy = self.forward(x)
            if MODEL_NAME == 'vit':
                outy = outy.logits
            y_hat_flat = torch.argmax(outy, dim=1)[:, None, :, :]
            y_hat_flats.append(y_hat_flat)

        for i in range(len(x)):
            y_hat_detach = [t.detach().cpu().numpy() for t in y_hat_flats]
            uncertainties.append(np.quantile(np.array(y_hat_detach).std(axis=0)[i], 0.95))

        return uncertainties


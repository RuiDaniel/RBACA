from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import torch
import random
import random
import yaml
import cv2
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
# from transformers import ViTFeatureExtractor
from torch.utils.data import DataLoader, TensorDataset

MODEL_NAME = 'vit' # options {'ResNet50', 'vit'}
FINAL_LAYER = 'Static' # options {'Static', 'Dynamic'}
BASE_RBACA_EVAL = 'B' # 'R' rbaca or 'B' base or 'E' eval


if MODEL_NAME == 'vit':
    ViT_IMAGE_DIM = 224
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

def inverse_norm01(r, min_x, max_x):
    # r is the normalized data (between 0 and 1)
    # min_x and max_x are the original minimum and maximum values of x
    original_range = max_x - min_x
    return r * original_range + min_x

def process_batch(batch, device):
    batch = [inverse_norm01(image, 0, 1) for image in batch]
    
    batch = torch.stack(batch)
    batch = batch.unsqueeze(1)
    batch = torch.cat([batch, batch, batch], dim=1)
    
    # batch_cpu = [image.cpu() for image in batch]

    # batch = np.array([image.numpy().transpose(1, 2, 0) for image in batch_cpu])
    # batch = torch.tensor(batch).permute(0, 3, 1, 2)#.to(device)

    # Process images using the ViT processor
    # processed_inputs = feature_extractor(images=batch, return_tensors="pt", do_normalize=True, do_rescale=False)
    processed_inputs = processor(images=batch, return_tensors="pt", do_normalize=True, do_rescale=False)
    
    batch = processed_inputs['pixel_values']
        
    return batch

def custom_transform(images, device, batch_size=64):
    # if DATA_AUG:
    #     batch_size = 4
            
    # Create a DataLoader for batching
    dataset = TensorDataset(torch.tensor(images))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # print(f'batch_size custom_transformBD = {batch_size}')
    
    all_processed_images = []
    
    for batch in dataloader:
        batch = batch[0]  # Extract the batch from the tuple
        # batch = batch.to(device)
        # print(f'Real batch_size custom_transformBD = {batch.shape[0]}')
        processed_batch = process_batch(batch, device)
        all_processed_images.append(processed_batch)
    
    # Concatenate all processed batches
    all_processed_images = torch.cat(all_processed_images, dim=0)
    
    return all_processed_images


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

class ContinuousDataset(Dataset):
    def init(self, datasetfile, transition_phase_after, order, seed, random=False):
        df = pd.read_csv(datasetfile)
        assert set(['train']).issubset(df.split.unique())
        np.random.seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if random:
            df = df.loc[df.split == 'train']
            df = df.sample(frac=1, random_state=seed)
            self.df = df
        else:
            res_dfs = []
            for r in order:
                res_df = df.loc[(df.scanner == r) & (df.split == 'train')]
                res_df = res_df.sample(frac=1, random_state=seed)
                res_dfs.append(res_df.reset_index(drop=True))

            combds = []  # Initialize the combds list

            new_idx = 0
            for j in range(len(res_dfs) - 1):
                old = res_dfs[j]
                new = res_dfs[j + 1]

                old_end = int((len(old) - new_idx) * transition_phase_after) + new_idx
                combds.append(old.iloc[new_idx:old_end].copy())

                old_idx = old_end
                old_max = len(old) - 1
                new_idx = 0
                i = 0

                while old_idx <= old_max and (i / ((old_max - old_end) * 2) < 1):
                    take_newclass = np.random.binomial(1, min(i / ((old_max - old_end) * 2), 1))
                    if take_newclass:
                        if new_idx < len(new):
                            combds.append(new.iloc[new_idx:new_idx + 1].copy())
                            new_idx += 1
                    else:
                        combds.append(old.iloc[old_idx:old_idx + 1].copy())
                        old_idx += 1
                    i += 1

                combds.append(old.iloc[old_idx:].copy())

            combds.append(new.iloc[new_idx:].copy())
            self.df = pd.concat(combds, ignore_index=True)

    def __len__(self):
        return len(self.df)



# def data_augmentation(image):
#     # Ensure image is in the range [0, 255] for cv2 operations
#     image_uint8 = (image * 255).astype(np.uint8)
    
#     # Rotation
#     rotated_180 = cv2.rotate(image_uint8, cv2.ROTATE_180)

#     # Flipping
#     flipped_horizontal = cv2.flip(image_uint8, 1)
#     flipped_vertical = cv2.flip(image_uint8, 0)

#     # Blurring
#     blurred = cv2.GaussianBlur(image_uint8, (15, 15), 0)

#     # Brightness change (increase and decrease by 10%)
#     brightness_increase = np.clip(image + 0.1, 0, 1)
#     brightness_decrease = np.clip(image - 0.1, 0, 1)

#     # # Gamma correction
#     # gamma = 0.3
#     # gamma_correction = np.array(255 * (image ** gamma), dtype='uint8')

#     # Contrast adjustment (increase and decrease by 10%)
#     contrast_increase = np.clip((image - 0.5) * 1.1 + 0.5, 0, 1)
#     contrast_decrease = np.clip((image - 0.5) * 0.9 + 0.5, 0, 1)

#     # Gaussian noise addition with reduced noise level
#     row, col = image.shape
#     mean = 0
#     var = 0.01
#     sigma = var ** 0.5
#     gaussian = np.random.normal(mean, sigma, (row, col))
#     noisy_image = np.clip(image + gaussian, 0, 1)

#     # Convert images back to [0, 1] range for display
#     rotated_180 = rotated_180 / 255.0
#     flipped_horizontal = flipped_horizontal / 255.0
#     flipped_vertical = flipped_vertical / 255.0
#     blurred = blurred / 255.0
#     # gamma_correction = gamma_correction / 255.0

#     # Display the results
#     images = np.stack([image, rotated_180, flipped_horizontal, flipped_vertical, blurred, brightness_increase, brightness_decrease, contrast_increase, contrast_decrease, noisy_image], axis=0)
    
#     # print(f'images.shape= {images.shape}')
    
#     plot_images(images)
#     return images

# def plot_images(images):
#     # Titles for the images
#     titles = [
#         'Original', 'Rotated 180', 'Flipped Horizontal', 'Flipped Vertical',
#         'Blurred', 'Brightness Increase', 'Brightness Decrease',
#         'Contrast Increase', 'Contrast Decrease', 'Noisy'
#     ]

#     # Create a figure to plot the images
#     fig, axes = plt.subplots(3, 4, figsize=(12, 9))

#     # Plot each image with its corresponding title
#     for i in range(10):
#         ax = axes.flat[i]
#         ax.imshow(images[i], cmap='gray')
#         ax.set_title(titles[i])
#         ax.axis('off')  # Hide axes

#     # Hide the last subplot as there are only 10 images
#     axes.flat[-1].axis('off')

#     # Save the plot to a .png file
#     plt.tight_layout()
#     plt.savefig('aug.png')
#     # plt.show()

def increase_labels(labels, n):
    result = []
    for element in labels:
        result.extend([element] * n)
    return result
    
    
class CardiacContinuous(ContinuousDataset):

    def __init__(self, datasetfile, transition_phase_after=.8, order=['Siemens', 'GE', 'Philips', 'Canon'], seed=None, random=False, model=None):
        super(ContinuousDataset, self).__init__()
        self.init(datasetfile, transition_phase_after, order, seed, random)

        self.outsize = (240, 192)
        
        self.model=model

    def crop_center_or_pad(self, img, cropx, cropy):
        x, y = img.shape

        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)

        if startx < 0:
            outimg = np.zeros(self.outsize)
            startx *= -1
            outimg[startx:self.outsize[0] - startx, :] = img[:, starty:starty + cropy]
            return outimg

        return img[startx:startx + cropx, starty:starty + cropy]

    def load_image(self, elem):
        img = np.load(elem.slicepath)

        if img.shape != self.outsize:
           img = self.crop_center_or_pad(img, self.outsize[0], self.outsize[1])
        
        # if DATA_AUG:
        #     images = data_augmentation(img)
        #     for i in range(len(images)):
        #         if MODEL_NAME == 'vit':
        #             images[i] = np.expand_dims(images[i], axis=0)
        #             images[i] = np.array(custom_transform(images[i], self.device))
        #             images[i] = np.squeeze(images[i], axis=0)
        #         else:
        #             images[i] = images[i][None, :, :]
        # else:
        images = img
        if MODEL_NAME == 'vit':
            images = np.expand_dims(images, axis=0)
            images = np.array(custom_transform(images, self.device))
            images = np.squeeze(images, axis=0)
        else:
            images = images[None, :, :]

        return images

    def __getitem__(self, index):
        elem = self.df.iloc[index]
        img = np.array(self.load_image(elem))
        
        label_disease = np.array([self.get_disease(elem.slicepath, self.model)])
        
        # print(f'img.shape = {img.shape}')
        # print(f'label_disease = {label_disease}')
        # print(f'len(label_disease) = {len(label_disease)}')
        
        # n = int(img.shape[0]/len(label_disease))
        # # print(f'confirming n=10={n}')
        # label_disease = increase_labels(label_disease, n)
        
        # return torch.as_tensor(img, dtype=torch.float32), torch.as_tensor(mask,
        #                                                                   dtype=torch.long), elem.scanner, elem.slicepath
        
        return torch.as_tensor(img, dtype=torch.float32), torch.as_tensor(label_disease, dtype=torch.int64), elem.scanner, elem.slicepath

    def get_disease(self, slicepath, model):
        excode = slicepath.split('/')[-2]
        dataset_info = pd.read_csv("../../Data/MM_cardiac/201014_M&Ms_Dataset_Information_-_opendataset.csv")
        filtered_row = dataset_info[dataset_info['External code'] == excode]
        if len(filtered_row) > 0:
            pathology = filtered_row.iloc[0]['Pathology']
            # print("Pathology for excode", excode, "is:", pathology)
        else:
            # print("No pathology found for excode", excode)
            pathology = 'Other'
            
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
        
        label = label_mapping[pathology]    
        
        return label
from datasets.BatchDataset import CardiacBatch
from datasets.ContinuousDataset import CardiacContinuous
import torch
from torch.utils.data import DataLoader
import pandas as pd
import yaml
import active_dynamicmemory.utils as admutils
import os
import active_dynamicmemory.runutils as rutils
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from monai.metrics import DiceMetric
import monai.networks.utils as mutils
import torch.nn.functional as F
import matplotlib.patches as mpatches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

INCLUDE_BACKGROUND_TESTING = True
INCLUDE_BACKGROUND_TRAINING = False
N_CLASSES = 4

def plot_segmented_images(model, params):
    
    
    splits = ['base', 'test', 'val', 'train']
    
    for split in splits:
        dl = DataLoader(CardiacBatch(params['trainparams']['datasetfile'], split=split), batch_size=16)
        
        for batch in dl:
            x_batch, y_batch, _, path = batch
            
            # print('path', path)
            
            x_batch = x_batch.to(device)
            y_pred_batch = model.forward(x_batch)  
            y_pred_batch = F.softmax(y_pred_batch, dim=1)
            _, y_pred_batch = y_pred_batch.max(dim=1)
            y_pred_batch = y_pred_batch.float()
        
        
            for k in range(len(y_pred_batch)):
                img = y_pred_batch[k]
                excode = path[k]
                target = y_batch[k]

                if excode.startswith('/home/catarinabarata/Data/MM_cardiac_pre_processed/B9H8N8/'):
                    print(excode)

                if excode == '/home/catarinabarata/Data/MM_cardiac_pre_processed/B9H8N8/0_3.npy':           
                    print('FOUND IT')
                    img = img[np.newaxis, :]
                    # predicted_labels = model_predict(model, img)

                    predicted_labels = img
                    
                    target = np.expand_dims(target, axis=0)
                    dsc, dice_classes = dice_mean_score(predicted_labels, target)

                    # Plot the specified slice
                    plt.figure(figsize=(6, 6))  # Adjust the figure size

                    # Extract the slice
                    slice_img = predicted_labels[0, :, :]

                    # Move tensor to CPU and convert to NumPy array
                    slice_img = slice_img.cpu().numpy()

                    # Plot the slice
                    plt.imshow(slice_img, cmap='gray')
                    plt.colorbar()
                    # plt.title('RBACA')
                    plt.title('RBACA')

                    # Add the Dice scores below the image
                    dice_text = (
                        f'Dice BG: {dice_classes[0]:.3f}, Dice LV: {dice_classes[1]:.3f}, Dice MYO: {dice_classes[2]:.3f}, Dice RV: {dice_classes[3]:.3f}'
                    )
                    plt.gcf().text(0.5, -0.1, dice_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5), transform=plt.gca().transAxes)

                    # Save the plot to a file
                    plt.savefig('B9H8N8_0_3.png')
                    plt.close()  # Close the figure to free memory

                    # Convert tensors to numpy arrays
                    predicted_labels = predicted_labels.cpu().numpy()
                    # target_np = target.cpu().numpy()

                    # Create an agreement/disagreement image
                    agreement = (predicted_labels == target)

                    # Initialize an RGB image
                    agreement_img = np.zeros((agreement.shape[1], agreement.shape[2], 3), dtype=np.uint8)

                    # Green for agreement, Red for disagreement
                    agreement_img[agreement[0, :, :]] = [0, 255, 0]  # Green
                    agreement_img[~agreement[0, :, :]] = [255, 0, 0]  # Red

                    # Plot the agreement/disagreement image
                    plt.figure(figsize=(6, 6))
                    plt.imshow(agreement_img)
                    plt.title('RBACA')

                    # Create a custom legend
                    green_patch = mpatches.Patch(color='green', label='Agreement')
                    red_patch = mpatches.Patch(color='red', label='Disagreement')
                    plt.legend(handles=[green_patch, red_patch], loc='upper right')

                    # Save the plot to a file
                    plt.savefig('B9H8N8_agreement.png')
                    plt.close()  # Close the figure to free memory
                    
                    print('done')
                    input()
                    
        
    
    return

def eval_cardiac(params, outfile, split='test'):
    """
    Base evaluation for cardiac on image level, stored to an outputfile
    """
    # print('params', params)
    device = torch.device('cuda')

    dl_test = DataLoader(CardiacBatch(params['trainparams']['datasetfile'], split=split), batch_size=16)
    model, _, _, _ = rutils.trained_model(params['trainparams'], params['settings'], training=False)

    model.to(device)
    model.eval()

    scanners = []
    dice_lv = []
    dice_myo = []
    dice_rv = []
    dice_bg = []
    dice_mean = []
    shifts = []
    img = []

    #final model
    # scanners_m, dice_lv_m, dice_myo_m, dice_rv_m, dice_mean_m, img_m = eval_cardiac_dl(model, dl_test)
    scanners_m, dice_mean_m, img_m, dice_bg_m, dice_lv_m, dice_myo_m, dice_rv_m  = eval_cardiac_dl(model, dl_test, params)
    scanners.extend(scanners_m)
    dice_lv.extend(dice_lv_m)
    dice_myo.extend(dice_myo_m)
    dice_rv.extend(dice_rv_m)
    dice_bg.extend(dice_bg_m)
    dice_mean.extend(dice_mean_m)
    img.extend(img_m)
    shifts.extend(['None']*len(scanners_m))

    modelpath = rutils.cached_path(params['trainparams'], params['settings']['TRAINED_MODELS_DIR'])
    #model shifts
    for s in params['trainparams']['order'][1:]:
        shiftmodelpath = f'{modelpath[:-3]}_shift_{s}.pt'
        model.model.load_state_dict(torch.load(shiftmodelpath, map_location=device))
        model.freeze()

        # scanners_m, dice_lv_m, dice_myo_m, dice_rv_m, dice_mean_m, img_m = eval_cardiac_dl(model, dl_test)
        scanners_m, dice_mean_m, img_m, dice_bg_m, dice_lv_m, dice_myo_m, dice_rv_m = eval_cardiac_dl(model, dl_test, params)
        scanners.extend(scanners_m)
        dice_lv.extend(dice_lv_m)
        dice_myo.extend(dice_myo_m)
        dice_rv.extend(dice_rv_m)
        dice_bg.extend(dice_bg_m)
        dice_mean.extend(dice_mean_m)
        img.extend(img_m)
        shifts.extend([s] * len(scanners_m))

    # df_results = pd.DataFrame({'scanner': scanners, 'dice_lv': dice_lv, 'dice_myo': dice_myo, 'dice_rv': dice_rv, 'dice_mean': dice_mean, 'shift': shifts})
    df_results = pd.DataFrame({'scanner': scanners, 'dice_mean': dice_mean, 'dice_lv': dice_lv, 'dice_myo': dice_myo, 'dice_rv': dice_rv, 'dice_bg': dice_bg, 'shift': shifts})
    # df_results = pd.DataFrame({'scanner': scanners, 'dice_mean': dice_mean, 'shift': shifts})
    # print('DF_RESULTS')
    # print(df_results)
    # df_results2 = pd.DataFrame({'dice_mean': dice_mean, 'dice_lv': dice_lv, 'dice_myo': dice_myo, 'dice_rv': dice_rv, 'dice_bg': dice_bg, 'img': img})

    df_results.to_csv(outfile, index=False)

def eval_cardiac_batch(params, outfile, split='test'):
    dl_test = DataLoader(CardiacBatch(params['trainparams']['datasetfile'], split=split), batch_size=16)
    model, _, _, _ = rutils.trained_model(params['trainparams'], params['settings'], training=False)

    scanners, dice_mean, img, dice_bg, dice_lv, dice_myo, dice_rv =  eval_cardiac_dl(model, dl_test, params)
    # scanners, dice_lv, dice_myo, dice_rv, dice_mean, img =  eval_cardiac_dl(model, dl_test)

    df_results = pd.DataFrame({'scanner': scanners, 'dice_mean': dice_mean, 'dice_lv': dice_lv, 'dice_myo': dice_myo, 'dice_rv': dice_rv, 'dice_bg': dice_bg})
    # df_results = pd.DataFrame({'scanner': scanners, 'dice_mean': dice_mean})
    df_results.to_csv(outfile, index=False)

def dice_mean_score(output, target):
    output = torch.tensor(output).to(device)
    target = torch.tensor(target).to(device)
    dice_scores = []
    dice_per_class = []
    
    if INCLUDE_BACKGROUND_TESTING:
        begin = 0
    else:
        begin = 1
        dice_per_class.append('None')

    for i in range(begin, N_CLASSES):
        # Create masks for output and target corresponding to the current class
        pred_class = torch.where(output == i, torch.ones_like(output), torch.zeros_like(output))
        target_class = torch.where(target == i, torch.ones_like(target), torch.zeros_like(target))
        intersection = torch.sum(pred_class * target_class)
        union = torch.sum(pred_class) + torch.sum(target_class)
        dice = (2. * intersection) / (union + 1e-8)  # Adding a small epsilon to avoid division by zero
        if union != 0:
            dice_scores.append(dice.item())
            dice_per_class.append(dice.item())
        else:
            dice_per_class.append('None')
            
    if len(dice_scores) == 0:
        dice_scores.append(1.0)

    mean_dice = torch.mean(torch.tensor(dice_scores))
    return mean_dice.item(), dice_per_class

def eval_cardiac_dl(model, dl, params, device='cuda'):
    device = torch.device(device)
    model.to(device)
    model.freeze()
    scanners = []
    dice_bg = []
    dice_lv = []
    dice_myo = []
    dice_rv = []
    dice_mean = []
    img = []
    
    plot_segmented_images(model, params)

    # dm = DiceMetric(include_background=False)

    for batch in dl:
        x_batch, y_batch, scanner, filepath = batch
        
        # print('x_batch.shape', x_batch.shape)
        # print('y_batch.shape', y_batch.shape)
        # print('scanner', scanner)
        # print('filepath', filepath)
        
        x_batch = x_batch.to(device)
        y_pred_batch = model.forward(x_batch)  
        y_pred_batch = F.softmax(y_pred_batch, dim=1)
        _, y_pred_batch = y_pred_batch.max(dim=1)
        y_pred_batch = y_pred_batch.float()
        
        
        
        # print('y_pred_batch.shape', y_pred_batch.shape)
        
        for i in range(len(x_batch)):
            y = y_batch[i]

            y_pred = y_pred_batch[i]
            y_true = y
            
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

            # print('y_pred', y_pred)
            # print('y_true', y_pred)
            # print('y_pred.shape', y_pred.shape)
            # print('y_true.shape', y_true.shape)
            
            # print('y_pred', y_pred)
            # print('y_true', y_true)

            dice, dice_per_class = dice_mean_score(y_pred, y_true)
            
            dice_bg.append(dice_per_class[0])
            dice_lv.append(dice_per_class[1])
            dice_myo.append(dice_per_class[2])
            dice_rv.append(dice_per_class[3])

            dice_mean.append(dice)
            img.append(filepath[i])
            scanners.append(scanner[i])

    return scanners, dice_mean, img, dice_bg, dice_lv, dice_myo, dice_rv

def eval_params(params, split='test'):
    # print('params1', params)
    settings = argparse.Namespace(**params['settings'])

    expname = admutils.get_expname(params['trainparams'])
    order = params['trainparams']['order'].copy()
    print(f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv')
    if not os.path.exists(f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv'):
        # print('entrei1')
        if params['trainparams']['continuous'] == False:
            # print('entrei2')
            eval_cardiac_batch(params, f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv', split=split)
        else:
            # print('entrei3')
            eval_cardiac(params, f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv', split=split)

    if params['trainparams']['continuous'] == False:
        print('entrei4')
        df = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv')
        df_temp = df.groupby(['scanner']).mean().reset_index()
        return df_temp

    df = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_{split}_dicescores.csv')
    
    # print('df')
    # print(df)
    
    df['shift'] = df['shift'].apply(lambda x: 'None' if pd.isna(x) else x)
    
#     print('df2')
#     print(df)
    
    df_temp = df.groupby(['scanner', 'shift']).mean().reset_index()
    
    # print('df_temp')
    # print(df_temp)

    df_res = df_temp.loc[df_temp['shift'] == 'None']
    df_bwt_fwt = df_temp.groupby(['scanner', 'shift']).mean().reset_index()
    # bwt = {'dice_lv': 0.0, 'dice_myo': 0.0, 'dice_rv': 0.0}
    # fwt = {'dice_lv': 0.0, 'dice_myo': 0.0, 'dice_rv': 0.0}
    bwt = {'dice_mean': 0.0}
    fwt = {'dice_mean': 0.0}
    
    # print('df_res')
    # print(df_res)
#     print('df_bwt_fwt')
#     print(df_bwt_fwt)
    
    # print('order')
    # print(order)

    for i in range(len(order) - 1):
        df_scanner = df_bwt_fwt.loc[df_bwt_fwt.scanner == order[i]]
        # print('df_scanner')
        # print(df_scanner)
        # bwt['dice_lv'] += df_scanner.loc[df_scanner['shift'] == 'None'].dice_lv.values[0] - \
        #                  df_scanner.loc[df_scanner['shift'] == order[i + 1]].dice_lv.values[0]
        # bwt['dice_myo'] += df_scanner.loc[df_scanner['shift'] == 'None'].dice_myo.values[0] - \
        #                  df_scanner.loc[df_scanner['shift'] == order[i + 1]].dice_myo.values[0]
        # bwt['dice_rv'] += df_scanner.loc[df_scanner['shift'] == 'None'].dice_rv.values[0] - \
        #                  df_scanner.loc[df_scanner['shift'] == order[i + 1]].dice_rv.values[0]
        
        bwt['dice_mean'] += df_scanner.loc[df_scanner['shift'] == 'None'].dice_mean.values[0] - \
                         df_scanner.loc[df_scanner['shift'] == order[i + 1]].dice_mean.values[0]

    order.append('None')
    
    df_ri = pd.read_csv("../Baseline2/dice_scores_bg_test.csv")
    random_init_data = df_ri.iloc[0, 1:6].to_dict()
    
    # print('random_init_data', random_init_data)

    for i in range(2, len(order)):
        df_scanner = df_bwt_fwt.loc[df_bwt_fwt.scanner == order[i - 1]]
        # fwt['dice_lv'] += df_scanner.loc[df_scanner['shift'] == order[i]].dice_lv.values[0] - \
        #                  df_scanner.loc[df_scanner['shift'] == order[1]].dice_lv.values[0]
        # fwt['dice_myo'] += df_scanner.loc[df_scanner['shift'] == order[i]].dice_myo.values[0] - \
        #                  df_scanner.loc[df_scanner['shift'] == order[1]].dice_myo.values[0]
        # fwt['dice_rv'] += df_scanner.loc[df_scanner['shift'] == order[i]].dice_rv.values[0] - \
        #                  df_scanner.loc[df_scanner['shift'] == order[1]].dice_rv.values[0]
        # print('df_scanner')
        # print(df_scanner)
        # # input()
        # print(order[i-2])
        # print(order[1])
        
        scanner_prev = order[i]
        
        if order[i-1] == 'Siemens':
            random_init = float(random_init_data['DSC Centre 1'])
        elif order[i-1] == 'Philips':
            random_init = float(random_init_data['DSC Centre 2']) * 0.5 + float(random_init_data['DSC Centre 3']) * 0.5
        elif order[i-1] == 'GE':
            random_init = float(random_init_data['DSC Centre 4'])
        else:
            random_init = float(random_init_data['DSC Centre 5'])
        
        # print('order[i-1]', order[i-1])
        # print('random_init', random_init)
        
        # fwt['dice_mean'] += df_scanner.loc[df_scanner['shift'] == scanner_prev].dice_mean.values[0] - \
        #                  df_scanner.loc[df_scanner['shift'] == random_init].dice_mean.values[0]
        fwt['dice_mean'] += df_scanner.loc[df_scanner['shift'] == scanner_prev].dice_mean.values[0]

    # bwt['dice_lv'] /= len(order) - 1
    # bwt['dice_myo'] /= len(order) - 1
    # bwt['dice_rv'] /= len(order) - 1
    bwt['dice_mean'] /= len(order) - 1

    # bwt['mean'] = (bwt['dice_lv']+bwt['dice_myo']+bwt['dice_rv'])/3

    # fwt['dice_lv'] /= len(order) - 1
    # fwt['dice_myo'] /= len(order) - 1
    # fwt['dice_rv'] /= len(order) - 1
    fwt['dice_mean'] /= len(order) - 1
    
    # df_res = df_res.append(pd.DataFrame({'scanner': ['BWT', 'FWT'], 'shift': ['None', 'None'],
    #                                      'dice_lv': [bwt['dice_lv'], fwt['dice_lv']],
    #                                      'dice_myo': [bwt['dice_myo'], fwt['dice_myo']],
    #                                      'dice_rv': [bwt['dice_rv'], fwt['dice_rv']]}))
    
    df_to_add = pd.DataFrame({'scanner': ['BWT', 'FWT'], 'shift': ['None', 'None'],
                                         'dice_mean': [bwt['dice_mean'], fwt['dice_mean']]})
#     print('df_res')
#     print(df_res)
    
#     print('df_to_add')
#     print(df_to_add)

    df_res = pd.concat([df_res, df_to_add], ignore_index=True)

    # print('df_res')
    # print(df_res)

    # df_res['mean'] = df_res.mean(axis=1)
    
    # print('df_res3')
    # print(df_res)

    return df_res

def eval_config(configfile, seeds=None, name=None, split='test'):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    if seeds is None:
        df = eval_params(params, split=split)
        if name is not None:
            df['model'] = name
        return df
    else:
        df = pd.DataFrame()
        for i, seed in enumerate(seeds):
            params['trainparams']['seed'] = seed
            params['trainparams']['run_postfix'] = seed
            df_temp = eval_params(params, split=split)
            df_temp['seed'] = seed
            if name is not None:
                df_temp['model'] = name
            df = pd.concat([df, df_temp], ignore_index=True)
        
        return df

def eval_config_list(configfiles, names, seeds=None, value='dice_mean'):
    assert type(configfiles) is list, 'configfiles should be a list'
    assert type(names) is list, 'method names should be a list'
    assert len(configfiles) == len(names), 'configfiles and method names should match'

    df_overall = pd.DataFrame()
    for k, configfile in enumerate(configfiles):
        df_conf = eval_config(configfile, seeds, names[k])
        df_overall = pd.concat([df_overall, df_conf], ignore_index=True)

    # print('df_overall')
    # print(df_overall)
        
    # if value!='std':
    df_overall['shift'] = df_overall['shift'].replace('None', 0)

    # print('df_overall')
    # print(df_overall)

    df_overview = df_overall.groupby(['model', 'scanner']).mean().reset_index()
    # print('df_overview')
    # print(df_overview)
    
    df_overall_filtered = df_overall[~df_overall['scanner'].isin(['BWT', 'FWT'])]
    df_overall_filtered = df_overall_filtered.drop(columns=['shift', 'model', 'scanner'])    
    df_overall_filtered = df_overall_filtered.groupby('seed').mean()
    df_overall_filtered['dice_mean'] = df_overall_filtered[['dice_lv', 'dice_myo', 'dice_rv', 'dice_bg']].mean(axis=1)

    
    # print(df_overall_filtered)
    # input()
    
    

    dice_bg_average = df_overall_filtered['dice_bg'].dropna().mean()
    dice_lv_average = df_overall_filtered['dice_lv'].dropna().mean()
    dice_myo_average = df_overall_filtered['dice_myo'].dropna().mean()
    dice_rv_average = df_overall_filtered['dice_rv'].dropna().mean()
    dice_mean_average = df_overall_filtered['dice_mean'].dropna().mean()
    
    
    
    new_row0 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_bg'], 'shift': [0.0], 
                     'dice_mean': [dice_bg_average], 'dice_lv': [None], 
                     'dice_myo': [None], 'dice_rv': [None], 
                     'dice_bg': [dice_bg_average], 'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row0], ignore_index=True)

    
    new_row1 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_lv'], 'shift': [0.0], 
                     'dice_mean': [dice_lv_average], 'dice_lv': [dice_lv_average], 
                     'dice_myo': [None], 'dice_rv': [None], 
                     'dice_bg': [None], 'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row1], ignore_index=True)

    
    new_row2 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_myo'], 'shift': [0.0], 
                     'dice_mean': [dice_myo_average], 'dice_lv': [None], 
                     'dice_myo': [dice_myo_average], 'dice_rv': [None], 
                     'dice_bg': [None], 'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row2], ignore_index=True)

    
    new_row3 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_rv'], 'shift': [0.0], 
                     'dice_mean': [dice_rv_average], 'dice_lv': [None], 
                     'dice_myo': [None], 'dice_rv': [dice_rv_average], 
                     'dice_bg': [None], 'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row3], ignore_index=True)
    
    
    
    new_row4 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_avg'], 'shift': [0.0], 
                     'dice_mean': [dice_mean_average], 'dice_lv': [None], 
                     'dice_myo': [None], 'dice_rv': [None], 
                     'dice_bg': [None], 'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row4], ignore_index=True)

    # print('df_overview2')
    # print(df_overview)

    df_overview = df_overview.pivot(index='model', columns='scanner', values=value).round(3)

    # print('df_overview')
    # print(df_overview)
        
    
    
    
    df_overview_std = df_overall.groupby(['model', 'scanner']).std().reset_index()
    
    # print('df_overview_std')
    # print(df_overview_std)
    
    
    dice_bg_average_std = df_overall_filtered['dice_bg'].dropna().std()
    dice_lv_average_std = df_overall_filtered['dice_lv'].dropna().std()
    dice_myo_average_std = df_overall_filtered['dice_myo'].dropna().std()
    dice_rv_average_std = df_overall_filtered['dice_rv'].dropna().std()
    dice_mean_average_std = df_overall_filtered['dice_mean'].dropna().std()
    
    new_row0_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_bg'], 'shift': [0.0], 
                     'dice_mean': [dice_bg_average_std], 'dice_lv': [None], 
                     'dice_myo': [None], 'dice_rv': [None], 
                     'dice_bg': [dice_bg_average_std], 'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row0_std], ignore_index=True)

    
    new_row1_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_lv'], 'shift': [0.0], 
                     'dice_mean': [dice_lv_average_std], 'dice_lv': [dice_lv_average_std], 
                     'dice_myo': [None], 'dice_rv': [None], 
                     'dice_bg': [None], 'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row1_std], ignore_index=True)

    
    new_row2_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_myo'], 'shift': [0.0], 
                     'dice_mean': [dice_myo_average_std], 'dice_lv': [None], 
                     'dice_myo': [dice_myo_average_std], 'dice_rv': [None], 
                     'dice_bg': [None], 'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row2_std], ignore_index=True)

    
    new_row3_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_rv'], 'shift': [0.0], 
                     'dice_mean': [dice_rv_average_std], 'dice_lv': [None], 
                     'dice_myo': [None], 'dice_rv': [dice_rv_average_std], 
                     'dice_bg': [None], 'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row3_std], ignore_index=True)
    
    
    
    new_row4_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_avg'], 'shift': [0.0], 
                     'dice_mean': [dice_mean_average_std], 'dice_lv': [None], 
                     'dice_myo': [None], 'dice_rv': [None], 
                     'dice_bg': [None], 'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row4_std], ignore_index=True)


    df_overview_std = df_overview_std.pivot(index='model', columns='scanner', values=value).round(3)
    
    # print('df_overview_std')
    # print(df_overview_std)
    
        
#     else:
#         df_overview = df_overall.groupby(['model', 'scanner']).std().reset_index()
        
#         dice_bg_average = df_overview['dice_bg'].dropna().mean()
#         new_row0 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_bg'], 'shift': [0.0], 
#                          'dice_mean': [dice_bg_average], 'dice_lv': [None], 
#                          'dice_myo': [None], 'dice_rv': [None], 
#                          'dice_bg': [dice_bg_average], 'seed': [1.0]})
#         df_overview = pd.concat([df_overview, new_row0], ignore_index=True)
        
#         dice_lv_average = df_overview['dice_lv'].dropna().mean()
#         new_row1 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_lv'], 'shift': [0.0], 
#                          'dice_mean': [dice_lv_average], 'dice_lv': [dice_lv_average], 
#                          'dice_myo': [None], 'dice_rv': [None], 
#                          'dice_bg': [None], 'seed': [1.0]})
#         df_overview = pd.concat([df_overview, new_row1], ignore_index=True)
        
#         dice_myo_average = df_overview['dice_myo'].dropna().mean()
#         new_row2 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_myo'], 'shift': [0.0], 
#                          'dice_mean': [dice_myo_average], 'dice_lv': [None], 
#                          'dice_myo': [dice_myo_average], 'dice_rv': [None], 
#                          'dice_bg': [None], 'seed': [1.0]})
#         df_overview = pd.concat([df_overview, new_row2], ignore_index=True)
        
#         dice_rv_average = df_overview['dice_rv'].dropna().mean()
#         new_row3 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['dice_rv'], 'shift': [0.0], 
#                          'dice_mean': [dice_rv_average], 'dice_lv': [None], 
#                          'dice_myo': [None], 'dice_rv': [dice_rv_average], 
#                          'dice_bg': [None], 'seed': [1.0]})
#         df_overview = pd.concat([df_overview, new_row3], ignore_index=True)
        
#         df_overview = df_overview.pivot(index='model', columns='scanner', values='mean').round(3)

    # print('df_overview')
    # print(df_overview)
    
    df_overview = df_overview.astype(str) + ' ± ' + df_overview_std.astype(str)

    return df_overview

def val_df_for_params(params):
    exp_name = admutils.get_expname(params['trainparams'])
    settings = argparse.Namespace(**params['settings'])

    max_version = max([int(x.split('_')[1]) for x in os.listdir(settings.LOGGING_DIR + exp_name)])
    df_temp = pd.read_csv(settings.LOGGING_DIR  + exp_name + '/version_{}/metrics.csv'.format(max_version))

    df_temp = df_temp.loc[df_temp['val_dice_lv_Canon'] == df_temp['val_dice_lv_Canon']]
    df_temp['idx'] = range(1, len(df_temp) + 1)

    return df_temp

def val_data_for_config(configfile, seeds=None):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    return val_data_for_params(params, seeds=seeds)

def val_data_for_params(params, seeds=None):
    df = pd.DataFrame()
    if seeds is None:
        df = df.append(val_df_for_params(params))
    else:
        for i, seed in enumerate(seeds):
            params['trainparams']['seed'] = seed
            params['trainparams']['run_postfix'] = i + 1
            df = df.append(val_df_for_params(params))

    for scanner in params['trainparams']['order']:
        # df[f'val_mean_{scanner}'] = (df[f'val_dice_lv_{scanner}'] + df[f'val_dice_rv_{scanner}'] + df[f'val_dice_myo_{scanner}']) / 3
        df[f'val_mean_{scanner}'] = df[f'val_dice_mean_{scanner}']

    return df

def plot_validation_curves(configfiles, val_measure='val_mean', names=None, seeds=None):
    
    assert type(configfiles) is list, "configfiles should be a list"

    fig, axes = plt.subplots(len(configfiles)+1, 1, figsize=(10, 2.5*(len(configfiles))))
    plt.subplots_adjust(hspace=0.0)

    for k, configfile in enumerate(configfiles):
        with open(configfile) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        df = val_data_for_params(params, seeds=seeds)
        ax = axes[k]
        for scanner in params['trainparams']['order']:
            sns.lineplot(data=df, y=f'{val_measure}_{scanner}', x='idx', ax=ax, label=scanner)
        ax.set_ylim(0.30, 0.80)
        #ax.set_yticks([0.85, 0.80, 0.75, 0.70])
        ax.get_xaxis().set_visible(False)
        ax.get_legend().remove()
        ax.set_xlim(1, df.idx.max())
        if names is not None:
            ax.set_ylabel(names[k])

        ax.tick_params(labelright=True, right=True)

    # creating timeline
    ds = CardiacContinuous(params['trainparams']['datasetfile'], seed=1)
    res = ds.df.scanner == params['trainparams']['order'][0]
    for j, s in enumerate(params['trainparams']['order'][1:]):
        res[ds.df.scanner == s] = j+2

    axes[-1].imshow(np.tile(res,(400,1)), cmap=ListedColormap(sns.color_palette()[:4]))
    axes[-1].get_yaxis().set_visible(False)
    axes[-1].get_yaxis()
from datasets.BatchDataset import CardiacBatch
# from datasets.ContinuousDataset import CardiacContinuous
import torch
from torch.utils.data import DataLoader
import pandas as pd
import yaml
import active_dynamicmemory.utils as admutils
import os
import active_dynamicmemory.runutils as rutils
import argparse
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
# from matplotlib.colors import ListedColormap
# import monai.networks.utils as mutils
import torch.nn.functional as F
from sklearn.metrics import f1_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

MODEL_NAME = 'ResNet50' # options {'ResNet50', 'vit'}
FINAL_LAYER = 'Dynamic' # options {'Static', 'Dynamic'}
BASE_RBACA_EVAL = 'B' # 'R' rbaca or 'B' base or 'E' eval



BATCH_SIZE = 16

# if DATA_AUG:
#     BATCH_SIZE = 1

BASE_MAPPING = [1, 2, 4, 0, -5, 6, -7, 3, 5]
RBACA_MAPPING = [1, 2, 4, 0, 8, 6, 7, 3, 5] # [2,3,0,1] if {0: 2, 1: 3, 2: 0, 3: 1}    
N_CLASSES = 9

classes_observed = set()
classes_observed_mapping = {}
for i in range(N_CLASSES):
    classes_observed_mapping[i] = -i - 1

def classes_observed_mapping_fill(new):
    for i in range(N_CLASSES):
        classes_observed_mapping[i] = new[i]
        
INIT_SIZE_DYNAMIC = 1

if BASE_RBACA_EVAL == 'R' and FINAL_LAYER == 'Dynamic':
    INIT_SIZE_DYNAMIC = len([x for x in BASE_MAPPING if x >= 0])
    classes_observed = set([i for i, x in enumerate(BASE_MAPPING) if x >= 0])
    classes_observed_mapping_fill(BASE_MAPPING)
elif BASE_RBACA_EVAL == 'E' and FINAL_LAYER == 'Dynamic':
    INIT_SIZE_DYNAMIC = len([x for x in RBACA_MAPPING if x >= 0])
    classes_observed = set([i for i, x in enumerate(RBACA_MAPPING) if x >= 0])
    classes_observed_mapping_fill(RBACA_MAPPING)
    
print(f'Eval: classes_observed = {classes_observed}')
print(f'Eval: classes_observed_mapping = {classes_observed_mapping}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def eval_cardiac(params, outfile, split='test'):
    """
    Base evaluation for cardiac on image level, stored to an outputfile
    """
    # print('params', params)
    device = torch.device('cuda')

    dl_test = DataLoader(CardiacBatch(params['trainparams']['datasetfile'], split=split), batch_size=BATCH_SIZE)
    # print(f'batch_size eval_cardiac = {BATCH_SIZE}')
    
    model, _, _, _ = rutils.trained_model(params['trainparams'], params['settings'], training=False)

    model.to(device)
    model.eval()

    scanners = []
    f1_mean = []
    f1_HCM = []
    f1_DCM = []
    f1_HHD = []
    f1_ARV = []
    f1_AHS = []
    f1_IHD = []
    f1_LVNC = []
    f1_NOR = []
    f1_Other = []
    filepaths = []
    
    shifts = []
    img = []

    #final model
    scanners_m, f1_mean_m, img_m, f1_HCM_m, f1_DCM_m, f1_HHD_m, f1_ARV_m, f1_AHS_m, f1_IHD_m, f1_LVNC_m, f1_NOR_m, f1_Other_m, filepaths_m = eval_cardiac_dl(model, dl_test)
    scanners.extend(scanners_m)
    f1_HCM.extend(f1_HCM_m)
    f1_DCM.extend(f1_DCM_m)
    f1_HHD.extend(f1_HHD_m)
    f1_ARV.extend(f1_ARV_m)
    f1_AHS.extend(f1_AHS_m)
    f1_IHD.extend(f1_IHD_m)
    f1_LVNC.extend(f1_LVNC_m)
    f1_NOR.extend(f1_NOR_m)
    f1_Other.extend(f1_Other_m)
    f1_mean.extend(f1_mean_m)
    img.extend(img_m)
    shifts.extend(['None']*len(scanners_m))
    filepaths.extend(filepaths_m)

    modelpath = rutils.cached_path(params['trainparams'], params['settings']['TRAINED_MODELS_DIR'])
    #model shifts
    for s in params['trainparams']['order'][1:]:
        shiftmodelpath = f'{modelpath[:-3]}_shift_{s}.pt'
        print(f'shiftmodelpath = {shiftmodelpath}')
        
        new_state_dict = torch.load(shiftmodelpath, map_location=device)
        
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

        scanners_m, f1_mean_m, img_m, f1_HCM_m, f1_DCM_m, f1_HHD_m, f1_ARV_m, f1_AHS_m, f1_IHD_m, f1_LVNC_m, f1_NOR_m, f1_Other_m, filepaths_m = eval_cardiac_dl(model, dl_test)
        scanners.extend(scanners_m)
        f1_HCM.extend(f1_HCM_m)
        f1_DCM.extend(f1_DCM_m)
        f1_HHD.extend(f1_HHD_m)
        f1_ARV.extend(f1_ARV_m)
        f1_AHS.extend(f1_AHS_m)
        f1_IHD.extend(f1_IHD_m)
        f1_LVNC.extend(f1_LVNC_m)
        f1_NOR.extend(f1_NOR_m)
        f1_Other.extend(f1_Other_m)
        f1_mean.extend(f1_mean_m)
        img.extend(img_m)
        shifts.extend([s] * len(scanners_m))
        filepaths.extend(filepaths_m)

    df_results = pd.DataFrame({'scanner': scanners, 'f1_mean': f1_mean, 'f1_HCM': f1_HCM, 'f1_DCM': f1_DCM, 
                               'f1_HHD': f1_HHD, 'f1_ARV': f1_ARV, 'f1_AHS': f1_AHS, 'f1_IHD': f1_IHD, 
                               'f1_LVNC': f1_LVNC, 'f1_NOR': f1_NOR, 'f1_Other': f1_Other, 'shift': shifts})
    # print('DF_RESULTS')
    # print(df_results)
    df_results.to_csv(outfile, index=False)

def eval_cardiac_batch(params, outfile, split='test'):
    dl_test = DataLoader(CardiacBatch(params['trainparams']['datasetfile'], split=split), batch_size=BATCH_SIZE)
    # print(f'batch_size eval_cardiac = {BATCH_SIZE}')
    
    model, _, _, _ = rutils.trained_model(params['trainparams'], params['settings'], training=False)

    scanners, f1_mean, img, f1_HCM, f1_DCM, f1_HHD, f1_ARV, f1_AHS, f1_IHD, f1_LVNC, f1_NOR, f1_Other, filepaths = eval_cardiac_dl(model, dl_test)

    df_results = pd.DataFrame({'scanner': scanners, 'f1_mean': f1_mean, 'f1_HCM': f1_HCM, 'f1_DCM': f1_DCM, 
                               'f1_HHD': f1_HHD, 'f1_ARV': f1_ARV, 'f1_AHS': f1_AHS, 'f1_IHD': f1_IHD, 
                               'f1_LVNC': f1_LVNC, 'f1_NOR': f1_NOR, 'f1_Other': f1_Other})

    df_results.to_csv(outfile, index=False)

def evaluate_f1_score(test_labels, predicted_labels):
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

def eval_cardiac_dl(model, dl, device='cuda'):
    device = torch.device(device)
    model.to(device)
    model.freeze()
    scanners = []
    f1_mean = []
    f1_HCM = []
    f1_DCM = []
    f1_HHD = []
    f1_ARV = []
    f1_AHS = []
    f1_IHD = []
    f1_LVNC = []
    f1_NOR = []
    f1_Other = []
    img = []
    filepaths = []


    for batch in dl:
        x_batch, y_batch, scanner, filepath = batch
        
        # print(f'Real batch_size eval_cardiac = {len(filepath)}')
        
        # print('eval_cardiac_dl: x_batch.shape: ', x_batch.shape, ', y_batch.shape', y_batch.shape)
        # print('scanner', scanner)
        # print('filepath', filepath)
        
        x_batch = x_batch.to(device)
        
        y_pred_batch = model.forward(x_batch)  # NEW
        if MODEL_NAME == 'vit':
            y_pred_batch = y_pred_batch.logits
        y_pred_batch = F.softmax(y_pred_batch, dim=1)
        _, y_pred_batch = y_pred_batch.max(dim=1)
        # y_pred_batch = y_pred_batch.float()
        
        
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
            
            if FINAL_LAYER == 'Dynamic':
                y_true = [classes_observed_mapping[t.item()] for t in y_true]
                y_true = torch.tensor(y_true, dtype=torch.long).to(device)
            
            # if filepath[i] == '/home/catarinabarata/Data/MM_cardiac_pre_processed/R1R6Y8/23_0.npy':
            #     print('y_pred', y_pred)
            #     print('y_true', y_true)
            
            f1_avg, f1_classes = evaluate_f1_score(y_true, y_pred)
            
            f1_HCM.append(f1_classes[0])
            f1_DCM.append(f1_classes[1])
            f1_HHD.append(f1_classes[2])
            f1_ARV.append(f1_classes[3])
            f1_AHS.append(f1_classes[4])
            f1_IHD.append(f1_classes[5])
            f1_LVNC.append(f1_classes[6])
            f1_NOR.append(f1_classes[7])
            f1_Other.append(f1_classes[8])

            f1_mean.append(f1_avg)
            img.append(filepath[i])
            scanners.append(scanner[i])
            filepaths.append(filepath)

    return scanners, f1_mean, img, f1_HCM, f1_DCM, f1_HHD, f1_ARV, f1_AHS, f1_IHD, f1_LVNC, f1_NOR, f1_Other, filepaths

def eval_params(params, split='test'):
    # print('params1', params)
    settings = argparse.Namespace(**params['settings'])

    expname = admutils.get_expname(params['trainparams'])
    order = params['trainparams']['order'].copy()
    print(f'{settings.RESULT_DIR}/cache/{expname}_{split}_f1scores.csv')
    if not os.path.exists(f'{settings.RESULT_DIR}/cache/{expname}_{split}_f1scores.csv'):
        # print('entrei1')
        if params['trainparams']['continuous'] == False:
            # print('entrei2')
            eval_cardiac_batch(params, f'{settings.RESULT_DIR}/cache/{expname}_{split}_f1scores.csv', split=split)
        else:
            # print('entrei3')
            eval_cardiac(params, f'{settings.RESULT_DIR}/cache/{expname}_{split}_f1scores.csv', split=split)

    if params['trainparams']['continuous'] == False:
        print('entrei4')
        df = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_{split}_f1scores.csv')
        df_temp = df.groupby(['scanner']).mean().reset_index()
        return df_temp

    df = pd.read_csv(f'{settings.RESULT_DIR}/cache/{expname}_{split}_f1scores.csv')
    
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
    bwt = {'f1_mean': 0.0}
    fwt = {'f1_mean': 0.0}
    
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
        
        bwt['f1_mean'] += df_scanner.loc[df_scanner['shift'] == 'None'].f1_mean.values[0] - \
                         df_scanner.loc[df_scanner['shift'] == order[i + 1]].f1_mean.values[0]

    order.append('None')
    
    # df_ri = pd.read_csv("../Baseline2/f1_scores_bg_test.csv")
    # random_init_data = df_ri.iloc[0, 1:6].to_dict()
    
    # print('random_init_data', random_init_data)

    for i in range(2, len(order)):
        df_scanner = df_bwt_fwt.loc[df_bwt_fwt.scanner == order[i - 1]]
        # print('df_scanner')
        # print(df_scanner)
        # # input()
        # print(order[i-2])
        # print(order[1])
        
        scanner_prev = order[i]
        
        # if order[i-1] == 'Siemens':
        #     random_init = float(random_init_data['DSC Centre 1'])
        # elif order[i-1] == 'Philips':
        #     random_init = float(random_init_data['DSC Centre 2']) * 0.5 + float(random_init_data['DSC Centre 3']) * 0.5
        # elif order[i-1] == 'GE':
        #     random_init = float(random_init_data['DSC Centre 4'])
        # else:
        #     random_init = float(random_init_data['DSC Centre 5'])
        
        # print('order[i-1]', order[i-1])
        # print('random_init', random_init)
        
        # fwt['f1_mean'] += df_scanner.loc[df_scanner['shift'] == scanner_prev].f1_mean.values[0] - \
        #                  df_scanner.loc[df_scanner['shift'] == random_init].f1_mean.values[0]
        fwt['f1_mean'] += df_scanner.loc[df_scanner['shift'] == scanner_prev].f1_mean.values[0]

    bwt['f1_mean'] /= len(order) - 1

    fwt['f1_mean'] /= len(order) - 1
    
    
    df_to_add = pd.DataFrame({'scanner': ['BWT', 'FWT'], 'shift': ['None', 'None'],
                                         'f1_mean': [bwt['f1_mean'], fwt['f1_mean']]})
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

def eval_config_list(configfiles, names, seeds=None, value='f1_mean'):
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
    df_overall_filtered['f1_mean'] = df_overall_filtered[['f1_HCM', 'f1_DCM', 'f1_HHD', 'f1_ARV', 'f1_AHS', 'f1_IHD', 'f1_LVNC', 'f1_NOR', 'f1_Other']].mean(axis=1)

    # print('df_overall_filtered')
    # print(df_overall_filtered)
    # input()
    
        
    f1_HCM_average = df_overall_filtered['f1_HCM'].dropna().mean()
    f1_DCM_average = df_overall_filtered['f1_DCM'].dropna().mean()
    f1_HHD_average = df_overall_filtered['f1_HHD'].dropna().mean()
    f1_ARV_average = df_overall_filtered['f1_ARV'].dropna().mean()
    f1_AHS_average = df_overall_filtered['f1_AHS'].dropna().mean()
    f1_IHD_average = df_overall_filtered['f1_IHD'].dropna().mean()
    f1_LVNC_average = df_overall_filtered['f1_LVNC'].dropna().mean()
    f1_NOR_average = df_overall_filtered['f1_NOR'].dropna().mean()
    f1_Other_average = df_overall_filtered['f1_Other'].dropna().mean()
    f1_mean_average = df_overall_filtered['f1_mean'].dropna().mean()
    
    
    
    new_row0 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_HCM'], 'shift': [0.0], 
                     'f1_mean': [f1_HCM_average], 'f1_HCM': [f1_HCM_average], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row0], ignore_index=True)
    
    new_row1 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_DCM'], 'shift': [0.0], 
                     'f1_mean': [f1_DCM_average], 'f1_HCM': [None], 
                     'f1_DCM': [f1_DCM_average], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row1], ignore_index=True)
    
    new_row2 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_HHD'], 'shift': [0.0], 
                     'f1_mean': [f1_HHD_average], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [f1_HHD_average], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row2], ignore_index=True)
    
    new_row3 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_ARV'], 'shift': [0.0], 
                     'f1_mean': [f1_ARV_average], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [f1_ARV_average], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row3], ignore_index=True)
    
    new_row4 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_AHS'], 'shift': [0.0], 
                     'f1_mean': [f1_AHS_average], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [f1_AHS_average], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row4], ignore_index=True)
    
    new_row5 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_IHD'], 'shift': [0.0], 
                     'f1_mean': [f1_IHD_average], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [f1_IHD_average], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row5], ignore_index=True)
    
    new_row6 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_LVNC'], 'shift': [0.0], 
                     'f1_mean': [f1_LVNC_average], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [f1_LVNC_average], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row6], ignore_index=True)
    
    new_row7 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_NOR'], 'shift': [0.0], 
                     'f1_mean': [f1_NOR_average], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [f1_NOR_average], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row7], ignore_index=True)
    
    new_row8 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_Other'], 'shift': [0.0], 
                     'f1_mean': [f1_Other_average], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [f1_Other_average], 
                     'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row8], ignore_index=True)
    
    new_row9 = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_avg'], 'shift': [0.0], 
                     'f1_mean': [f1_mean_average], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [f1_Other_average], 
                     'seed': [1.0]})
    df_overview = pd.concat([df_overview, new_row9], ignore_index=True)

    

    # print('df_overview2')
    # print(df_overview)

    df_overview = df_overview.pivot(index='model', columns='scanner', values=value).round(3)

    # print('df_overview')
    # print(df_overview)
        
    
    
    
    df_overview_std = df_overall.groupby(['model', 'scanner']).std().reset_index()
    
    # print('df_overview_std')
    # print(df_overview_std)
    
    
    f1_HCM_average_std = df_overall_filtered['f1_HCM'].dropna().std()
    f1_DCM_average_std = df_overall_filtered['f1_DCM'].dropna().std()
    f1_HHD_average_std = df_overall_filtered['f1_HHD'].dropna().std()
    f1_ARV_average_std = df_overall_filtered['f1_ARV'].dropna().std()
    f1_AHS_average_std = df_overall_filtered['f1_AHS'].dropna().std()
    f1_IHD_average_std = df_overall_filtered['f1_IHD'].dropna().std()
    f1_LVNC_average_std = df_overall_filtered['f1_LVNC'].dropna().std()
    f1_NOR_average_std = df_overall_filtered['f1_NOR'].dropna().std()
    f1_Other_average_std = df_overall_filtered['f1_Other'].dropna().std()
    f1_mean_average_std = df_overall_filtered['f1_mean'].dropna().std()

    new_row0_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_HCM'], 'shift': [0.0], 
                     'f1_mean': [f1_HCM_average_std], 'f1_HCM': [f1_HCM_average_std], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row0_std], ignore_index=True)
    
    new_row1_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_DCM'], 'shift': [0.0], 
                     'f1_mean': [f1_DCM_average_std], 'f1_HCM': [None], 
                     'f1_DCM': [f1_DCM_average_std], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row1_std], ignore_index=True)
    
    new_row2_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_HHD'], 'shift': [0.0], 
                     'f1_mean': [f1_HHD_average_std], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [f1_HHD_average_std], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row2_std], ignore_index=True)
    
    new_row3_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_ARV'], 'shift': [0.0], 
                     'f1_mean': [f1_ARV_average_std], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [f1_ARV_average_std], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row3_std], ignore_index=True)
    
    new_row4_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_AHS'], 'shift': [0.0], 
                     'f1_mean': [f1_AHS_average_std], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [f1_AHS_average_std], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row4_std], ignore_index=True)
    
    new_row5_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_IHD'], 'shift': [0.0], 
                     'f1_mean': [f1_IHD_average_std], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [f1_IHD_average_std], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row5_std], ignore_index=True)
    
    new_row6_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_LVNC'], 'shift': [0.0], 
                     'f1_mean': [f1_LVNC_average_std], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [f1_LVNC_average_std], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row6_std], ignore_index=True)
    
    new_row7_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_NOR'], 'shift': [0.0], 
                     'f1_mean': [f1_NOR_average_std], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [f1_NOR_average_std], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row7_std], ignore_index=True)
    
    new_row8_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_Other'], 'shift': [0.0], 
                     'f1_mean': [f1_Other_average_std], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [f1_Other_average_std], 
                     'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row8_std], ignore_index=True)
    
    new_row9_std = pd.DataFrame({'model': ['RBACA'], 'scanner': ['f1_avg'], 'shift': [0.0], 
                     'f1_mean': [f1_mean_average_std], 'f1_HCM': [None], 
                     'f1_DCM': [None], 'f1_HHD': [None], 
                     'f1_ARV': [None], 'f1_AHS': [None], 
                     'f1_IHD': [None], 'f1_LVNC': [None], 
                     'f1_NOR': [None], 'f1_Other': [None], 
                     'seed': [1.0]})
    df_overview_std = pd.concat([df_overview_std, new_row9_std], ignore_index=True)
    


    df_overview_std = df_overview_std.pivot(index='model', columns='scanner', values=value).round(3)
    
    # print('df_overview_std')
    # print(df_overview_std)

    # print('df_overview')
    # print(df_overview)
    
    df_overview = df_overview.astype(str) + ' ± ' + df_overview_std.astype(str)

    return df_overview

def val_df_for_params(params):
    exp_name = admutils.get_expname(params['trainparams'])
    settings = argparse.Namespace(**params['settings'])

    max_version = max([int(x.split('_')[1]) for x in os.listdir(settings.LOGGING_DIR + exp_name)])
    df_temp = pd.read_csv(settings.LOGGING_DIR  + exp_name + '/version_{}/metrics.csv'.format(max_version))

    # df_temp = df_temp.loc[df_temp['val_f1_avg_Canon'] == df_temp['val_f1_avg_Canon']] ???
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
        df[f'val_mean_{scanner}'] = df[f'val_f1_mean_{scanner}']

    return df

# def plot_validation_curves(configfiles, val_measure='val_mean', names=None, seeds=None):
#     assert type(configfiles) is list, "configfiles should be a list"

#     fig, axes = plt.subplots(len(configfiles)+1, 1, figsize=(10, 2.5*(len(configfiles))))
#     plt.subplots_adjust(hspace=0.0)

#     for k, configfile in enumerate(configfiles):
#         with open(configfile) as f:
#             params = yaml.load(f, Loader=yaml.FullLoader)

#         df = val_data_for_params(params, seeds=seeds)
#         ax = axes[k]
#         for scanner in params['trainparams']['order']:
#             sns.lineplot(data=df, y=f'{val_measure}_{scanner}', x='idx', ax=ax, label=scanner)
#         ax.set_ylim(0.30, 0.80)
#         #ax.set_yticks([0.85, 0.80, 0.75, 0.70])
#         ax.get_xaxis().set_visible(False)
#         ax.get_legend().remove()
#         ax.set_xlim(1, df.idx.max())
#         if names is not None:
#             ax.set_ylabel(names[k])

#         ax.tick_params(labelright=True, right=True)

#     # creating timeline
#     ds = CardiacContinuous(params['trainparams']['datasetfile'], seed=1)
#     res = ds.df.scanner == params['trainparams']['order'][0]
#     for j, s in enumerate(params['trainparams']['order'][1:]):
#         res[ds.df.scanner == s] = j+2

#     axes[-1].imshow(np.tile(res,(400,1)), cmap=ListedColormap(sns.color_palette()[:4]))
#     axes[-1].get_yaxis().set_visible(False)
#     axes[-1].get_yaxis()
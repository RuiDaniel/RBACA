from glob import glob
import os
import pandas as pd
import nibabel as nib
import numpy as np
import argparse
import dicom2nifti

def norm01(x):
    r = (x - np.min(x))
    m = np.max(r)
    if m > 0:
        r = np.divide(r, np.max(r))
    return r

def convert_nifti_to_dicom(nifti_path, output_path):
    dicom2nifti.convert_directory(nifti_path, output_path, compression=True, reorient=True)

def prepare_data(datasetpath, outputpath, seed=516165):
    df_dsinfo = pd.read_csv(f'{datasetpath}/201014_M&Ms_Dataset_Information_-_opendataset.csv')
    df_dsinfo['labels'] = True

    # for p in glob(f'{datasetpath}/Training/Unlabeled/*'):
    #     df_dsinfo.loc[df_dsinfo['External code'] == p.split('/')[-1], 'labels'] = False

    df_dsinfo = df_dsinfo.loc[df_dsinfo.labels]
    df_base = df_dsinfo.loc[df_dsinfo.VendorName == 'Siemens']
    df_base = df_base.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_base['split'] = 'None'
    df_base.loc[0:49, 'split'] = 'base'
    df_base.loc[50:74, 'split'] = 'train'
    df_base.loc[75:84, 'split'] = 'val'
    df_base.loc[85:, 'split'] = 'test'

    df_phil = df_dsinfo.loc[df_dsinfo.VendorName == 'Philips']
    df_phil = df_phil.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_phil['split'] = 'None'
    df_phil.loc[0:104, 'split'] = 'train'
    df_phil.loc[105:114, 'split'] = 'val'
    df_phil.loc[115:, 'split'] = 'test'

    df_ge = df_dsinfo.loc[df_dsinfo.VendorName == 'GE']
    df_ge = df_ge.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_ge['split'] = 'None'
    df_ge.loc[0:29, 'split'] = 'train'
    df_ge.loc[30:39, 'split'] = 'val'
    df_ge.loc[40:, 'split'] = 'test'

    df_canon = df_dsinfo.loc[df_dsinfo.VendorName == 'Canon']
    df_canon = df_canon.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_canon['split'] = 'None'
    df_canon.loc[0:29, 'split'] = 'train'
    df_canon.loc[30:39, 'split'] = 'val'
    df_canon.loc[40:, 'split'] = 'test'

    df_train = pd.concat([df_base, df_phil, df_ge, df_canon])
    df_train.groupby(['VendorName', 'split']).count()

    scanners = []
    imgs = []
    ts = []
    slices = []
    splits = []
    slicepath = []
    centres = []
    
    print(len(df_train))
    for index, row in df_train.iterrows():
        if index % 20 == 0:
            print(index)
            print("Length of scanners:", len(scanners))
            print("Length of imgs:", len(imgs))
            print("Length of slicepath:", len(slicepath))
            print("Length of ts:", len(ts))
            print("Length of slices:", len(slices))
            print("Length of splits:", len(splits))
            
        excode = row['External code']

        if os.path.exists(f'{datasetpath}/Training/Labeled/{excode}/'):
            filepath = f'{datasetpath}/Training/Labeled/{excode}/{excode}_sa.nii.gz'
            maskpath = f'{datasetpath}/Training/Labeled/{excode}/{excode}_sa_gt.nii.gz'
        elif os.path.exists(f'{datasetpath}/Testing/{excode}/'):
            filepath = f'{datasetpath}/Testing/{excode}/{excode}_sa.nii.gz'
            maskpath = f'{datasetpath}/Testing/{excode}/{excode}_sa_gt.nii.gz'
        elif os.path.exists(f'{datasetpath}/Validation/{excode}/'):
            filepath = f'{datasetpath}/Validation/{excode}/{excode}_sa.nii.gz'
            maskpath = f'{datasetpath}/Validation/{excode}/{excode}_sa_gt.nii.gz'
        

        img = nib.load(filepath)
        imgd = img.get_fdata()
        mask = nib.load(maskpath)
        maskd = mask.get_fdata()
        sl = img.shape[2]

        slices.extend(list(range(0, sl)))
        imgs.extend([filepath] * sl)
        ts.extend([row.ED] * sl)
        scanners.extend([row.VendorName] * sl)
        splits.extend([row.split] * sl)
        centres.extend([row.Centre] * sl)
    
        # Inside the loop where you save the files
        for i in range(0, sl):
            # Extracting directory path
            directory = f'{outputpath}/{excode}'
            # Creating directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)

            if i < imgd.shape[2] and row.ED < imgd.shape[3]:
                img_sl = imgd[:, :, i, row.ED]
                img_sl = norm01(img_sl)
                np.save(f'{directory}/{row.ED}_{i}.npy', img_sl)
                slicepath.append(f'{directory}/{row.ED}_{i}.npy')
                mask_sl = maskd[:, :, i, row.ED]
                # mask_sl = norm01(mask_sl)  # Apply normalization to mask
                np.save(f'{directory}/{row.ED}_{i}_gt.npy', mask_sl)
            else:
                slicepath.append('Error')

                

        slices.extend(list(range(0, sl)))
        imgs.extend([filepath] * sl)
        ts.extend([row.ES] * sl)
        scanners.extend([row.VendorName] * sl)
        splits.extend([row.split] * sl)
        centres.extend([row.Centre] * sl)

        for i in range(0, sl):
            # Extracting directory path
            directory = f'{outputpath}/{excode}'
            # Creating directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            if i < imgd.shape[2] and row.ES < imgd.shape[3]:
                img_sl = imgd[:, :, i, row.ES]
                img_sl = norm01(img_sl)
                np.save(f'{outputpath}/{excode}/{row.ES}_{i}.npy', img_sl)
                slicepath.append(f'{outputpath}/{excode}/{row.ES}_{i}.npy')
                mask_sl = maskd[:, :, i, row.ES]
                # mask_sl = norm01(mask_sl)  # Apply normalization to mask
                np.save(f'{outputpath}/{excode}/{row.ES}_{i}_gt.npy', mask_sl)
            else:
                slicepath.append('Error')
                
                
    print("Length of scanners:", len(scanners))
    print("Length of centres:", len(centres))
    # print(scanners)
    print("Length of imgs:", len(imgs))
    # print(imgs)
    print("Length of slicepath:", len(slicepath))
    # print(slicepath)
    print("Length of ts:", len(ts))
    # print(ts)
    print("Length of slices:", len(slices))
    # print(slices)
    print("Length of splits:", len(splits))
    # print(splits)
    


    df_train_slices = pd.DataFrame({'scanner': scanners, 'centre': centres, 'filepath': imgs, 'slicepath': slicepath, 't': ts, 'slice': slices, 'split': splits})
    
    df_train_slices = df_train_slices.dropna(subset=['slicepath'])
    
    df_train_slices.to_csv(f'{outputpath}/cardiacdatasetsplit.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare the cardiac segmentation dataset for dynamic memory.')
    parser.add_argument('datasetpath', type=str, help='path to the downloaded M&Ms Challenge dataset')
    parser.add_argument('outputpath', type=str, help='path to the directory where the prepared dataframe and single slices are stored')
    args = parser.parse_args()

    prepare_data(os.path.abspath(args.datasetpath), os.path.abspath(args.outputpath))
    # prepare_data('MM_cardiac', 'MM_cardiac_pre_processed2')

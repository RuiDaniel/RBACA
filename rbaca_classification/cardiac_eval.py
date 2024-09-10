import sys
sys.path.append('../')
from cardiac_evalfunctions import *
import active_dynamicmemory.runutils as rutils
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


df_overview = eval_config_list(['training_configs/cardiac_rbaca.yml'],
                               ['RBACA'], seeds=[1,2,3])

df_selected = df_overview[['Siemens', 'GE', 'Philips', 'Canon', 'BWT', 'FWT', 'f1_avg', 
                           'f1_HCM', 'f1_DCM', 'f1_HHD', 'f1_ARV', 'f1_AHS', 'f1_IHD', 'f1_LVNC', 'f1_NOR', 'f1_Other']]

df_selected = df_selected.rename(columns={'Siemens': 'F1 Siemens', 'GE': 'F1 GE', 'Philips': 'F1 Philips', 'Canon': 'F1 Canon'})
    
print('df_selected')
print(df_selected)

df_selected.to_csv(`results.csv`, index=False)


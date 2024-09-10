import sys
sys.path.append('../')
from cardiac_evalfunctions import *
import active_dynamicmemory.runutils as rutils
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

df_overview = eval_config_list(['training_configs/cardiac_rbaca.yml'],
                               ['RBACA'], seeds=[1,2,3])

df_selected = df_overview[['Siemens', 'GE', 'Philips', 'Canon', 'BWT', 'FWT', 'dice_avg', 'dice_bg',  'dice_lv',  'dice_myo',  'dice_rv']]

print('df_selected')
print(df_selected)

df_selected.to_csv(`results.csv`, index=False)


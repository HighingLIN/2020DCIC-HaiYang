# %%
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')



# %%
type_dict = {'围网':'wei','拖网':'tuo','刺网':'ci'}


# %%
train = pd.read_hdf('./train.h5')


# %%
t = train[train['ship']==2963]


# %%
plt.plot(t['v'])


# %%
plt.plot(t['d'])


# %%
def show_path(type_name):
    ids = train[train['type']==type_name]['ship'].unique()
    ids = rn.sample(list(ids),10)
    t = train[train['ship'].isin(ids)]

    f, ax = plt.subplots(5,2, figsize=(8,20))
    for index, cur_id in enumerate(ids):
        cur = t[t['ship']==cur_id]
        i = index // 2
        j = index % 2
        ax[i,j].plot(cur['x'], cur['y'])
        ax[i,j].set_title(cur_id)


# %%
train[train['ship']==2963]


# %%
show_path('围网')


# %%
train[train['ship']==4022]


# %%

show_path('拖网')


# %%
show_path('刺网')


# %%
train[train['ship']==1415]






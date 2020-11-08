# %%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
#%%
warnings.filterwarnings('ignore')
os.chdir('E:\Python\Machine')
train_path = './hy_round1_train_20200102'
test_path = './hy_round1_testA_20200102'


# %%
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
print(len(train_files)," ", len(test_files))



# %%
df = pd.read_csv(f'{train_path}/6966.csv')


# %%
df.head()


# %%
df['type'].unique()


# %%
df.shape

# %%
ret = []
for file in tqdm(train_files):
    df = pd.read_csv(f'{train_path}/{file}')
    ret.append(df)
df = pd.concat(ret)
df.columns = ['ship','x','y','v','d','time','type']


# %%
df.to_hdf('./train.h5', 'df', mode='w')


# %%
ret = []
for file in tqdm(test_files):
    df = pd.read_csv(f'{test_path}/{file}')
    ret.append(df)
df = pd.concat(ret)
df.columns = ['ship','x','y','v','d','time']


# %%
df.to_hdf('./test.h5', 'df', mode='w')




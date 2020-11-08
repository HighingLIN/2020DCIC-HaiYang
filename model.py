#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.integrate import quad
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import time
import datetime
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小


pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')
os.chdir('E:\Python\Machine')


# %%
# 导入
train = pd.read_hdf('./train.h5')
train.sort_values(by=['ship','time'],inplace=True)
train.reset_index(drop=True,inplace=True)
#%%
test = pd.read_hdf('./test.h5')
test.sort_values(by=['ship','time'],inplace=True)
test.reset_index(drop=True,inplace=True)


#%%
# 缺失值分析
def missing_values(df):
    alldata_na = pd.DataFrame(df.isnull().sum(), columns={'missingNum'})
    alldata_na['existNum'] = len(df) - alldata_na['missingNum']
    alldata_na['sum'] = len(df)
    alldata_na['missingRatio'] = alldata_na['missingNum']/len(df)*100
    alldata_na['dtype'] = df.dtypes
    #ascending：默认True升序排列；False降序排列
    alldata_na = alldata_na[alldata_na['missingNum']>0].reset_index().sort_values(by=['missingNum','index'],ascending=[False,True])
    alldata_na.set_index('index',inplace=True)
    return alldata_na



#%%
# 可视化图
def box(df):
    plt.figure(figsize=(10,5))
    sns.boxplot(df)
    plt.show()

def sca(df):
    sns.lmplot('x','y',df,hue='type', fit_reg=False)



# %%
#分割时间
def diff_time(df):
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    return df
#%%
train = diff_time(train)
#%%
test = diff_time(test)



# %%
#清除重复值
def drop_hu(df):
    temp=df.drop_duplicates(subset=['ship','month','day','hour','minute'])
    return temp
#%%
train=drop_hu(train)
train.reset_index(drop=True,inplace=True)
#%%
test=drop_hu(test)
test.reset_index(drop=True,inplace=True)



#%%
# IF清除异常值
def IFdrop(df):
    IForest = IsolationForest(random_state=0,n_jobs=-1,verbose=0)
    y_pred = IForest.fit_predict(df.values.reshape(-1,1))
    return pd.DataFrame(y_pred)

def IFdrop1(df):
    IForest = IsolationForest(random_state=0,n_jobs=-1,verbose=0,contamination=0.03)
    y_pred = IForest.fit_predict(df.values.reshape(-1,2))
    return pd.DataFrame(y_pred)

#%%
temp=train.groupby('ship')['x'].apply(IFdrop)
drop_index=train.loc[(temp.values==-1).reshape(-1,)].index
train=train.drop(drop_index)
train.reset_index(drop=True,inplace=True)
#%%
temp=train.groupby('ship')['v'].apply(IFdrop)
drop_index=train.loc[(temp.values==-1).reshape(-1,)].index
train=train.drop(drop_index)
train.reset_index(drop=True,inplace=True)
#%%
temp=train.groupby('ship')[['x','y']].apply(IFdrop1)
drop_index=train.loc[(temp.values==-1).reshape(-1,)].index
train=train.drop(drop_index)
train.reset_index(drop=True,inplace=True)
#%%
temp=test.groupby('ship')['x'].apply(IFdrop)
drop_index=test.loc[(temp.values==-1).reshape(-1,)].index
test=test.drop(drop_index)
test.reset_index(drop=True,inplace=True)
#%%
temp=test.groupby('ship')['v'].apply(IFdrop)
drop_index=test.loc[(temp.values==-1).reshape(-1,)].index
test=test.drop(drop_index)
test.reset_index(drop=True,inplace=True)
#%%
temp=test.groupby('ship')[['x','y']].apply(IFdrop1)
drop_index=test.loc[(temp.values==-1).reshape(-1,)].index
test=test.drop(drop_index)
test.reset_index(drop=True,inplace=True)




#%%
# DBSCAN清除异常值
def dbscan(df):
    outlier_detection = DBSCAN(n_jobs=-1)
    y_pred = pd.DataFrame(outlier_detection.fit_predict(df))




#%%
# 填充值1
def is_fill(col):
    a=np.zeros((col.shape[0],),dtype=np.int)
    for i in range(col.shape[0]-1):
        if(col.iloc[i,0]==col.iloc[i+1,0]):
            if(col.iloc[i,1]==col.iloc[i+1,1]):
                a[i]=0
            else:
                a[i]=1
        else:
            a[i]=1
    a[-1]=1
    return pd.DataFrame(a)
#%%
i=train.groupby('ship')['x','y'].apply(is_fill).reset_index(drop=True).rename({0:'fill'},axis=1)
train=pd.concat([train,i],axis=1)
#%%
i=test.groupby('ship')['x','y'].apply(is_fill).reset_index(drop=True).rename({0:'fill'},axis=1)
test=pd.concat([test,i],axis=1)
#%%
def fill_values(col):
    a=np.zeros((col.shape[0],))
    for i in range(col.shape[0]):
        if(col.iloc[i,0]==0):
            a[i]=0
        else:
            a[i]=col.iloc[i,1]
    return pd.DataFrame(a,columns=['fill_v'])
#%%
i=train.groupby('ship')['fill','v'].apply(fill_values).reset_index(drop=True)
train=pd.concat([train,i],axis=1)
train=train.drop(['v'],axis=1)
train.rename({'fill_v':'v'},axis=1,inplace=True)
#%%
i=test.groupby('ship')['fill','v'].apply(fill_values).reset_index(drop=True)
test=pd.concat([test,i],axis=1)
test=test.drop(['v'],axis=1)
test.rename({'fill_v':'v'},axis=1,inplace=True)
#%%
# 填充值2
def is_fill(col):
    a=np.zeros((col.shape[0],),dtype=np.int)
    for i in range(col.shape[0]):
        if(i==0):
            if((col.iloc[i,0]==col.iloc[i+1,0])and(col.iloc[i,1]==col.iloc[i+1,1])):
                a[i]=0
            else:
                a[i]=1
        elif(i<col.shape[0]-1):
            if((col.iloc[i,0]==col.iloc[i+1,0])and(col.iloc[i,0]==col.iloc[i-1,0])):
                if((col.iloc[i,1]==col.iloc[i+1,1])and(col.iloc[i,1]==col.iloc[i-1,1])):
                    a[i]=0
                else:
                    a[i]=1
            else:
                a[i]=1
        else:
            if((col.iloc[i,0]==col.iloc[i-1,0])and(col.iloc[i,1]==col.iloc[i-1,1])):
                a[i]=0
            else:
                a[i]=1
    
    return pd.DataFrame(a)
#%%
i=train.groupby('ship')['x','y'].apply(is_fill).reset_index(drop=True).rename({0:'fill'},axis=1)
train=pd.concat([train,i],axis=1)
#%%
i=test.groupby('ship')['x','y'].apply(is_fill).reset_index(drop=True).rename({0:'fill'},axis=1)
test=pd.concat([test,i],axis=1)
#%%
def fill_values(col):
    a=np.zeros((col.shape[0],2))
    for i in range(col.shape[0]):
        if(col.iloc[i,0]==0):
            a[i,0]=0
            a[i,1]=0
        else:
            a[i,0]=col.iloc[i,1]
            a[i,1]=col.iloc[i,2]
    return pd.DataFrame(a,columns=['fill_v','fill_d'])
#%%
i=train.groupby('ship')['fill','v','d'].apply(fill_values).reset_index(drop=True)
train=pd.concat([train,i],axis=1)
train=train.drop(['v','d'],axis=1)
train.rename({'fill_v':'v','fill_d':'d'},axis=1,inplace=True)
#%%
i=test.groupby('ship')['fill','v','d'].apply(fill_values).reset_index(drop=True)
test=pd.concat([test,i],axis=1)
test=test.drop(['v','d'],axis=1)
test.rename({'fill_v':'v','fill_d':'d'},axis=1,inplace=True)




#%%
# 日最大最小差
def maxminchange(col):
    a=col.max()-col.min()
    return a
# 天数换算
def day(col):
    a=np.zeros((col.shape[0]),np.int)
    for i in range(col.shape[0]):
        a[i]=i+1
    return pd.DataFrame(a)
# 日变化
def lafichange(col):
    a=col.iloc[-1,0]-col.iloc[0,0]
    return a
#%%
t=train.groupby(['ship','day'])[['x']].apply(maxminchange).reset_index().rename({'x':'dx'},axis=1)
i=t.groupby('ship')['day'].apply(day).reset_index().rename({0:'day1'},axis=1)
i.drop(['ship','level_1'],axis=1,inplace=True)
ti=pd.concat([t,i],axis=1)
ti.drop('day',axis=1,inplace=True)
t1=pd.pivot(ti,index='ship',columns='day1',values='dx').reset_index().rename({1:'day1_x',2:'day2_x',3:'day3_x',4:'day4_x'},axis=1)
t1.loc[t1['day1_x']==0.,'day1_x']=1e-8
t1.loc[t1['day2_x']==0.,'day2_x']=1e-8
t1.loc[t1['day3_x']==0.,'day3_x']=1e-8
t1.loc[t1['day4_x']==0.,'day4_x']=1e-8
t11=t1.fillna(-1)
#%%
t=test.groupby(['ship','day'])[['x']].apply(maxminchange).reset_index().rename({'x':'dx'},axis=1)
i=t.groupby('ship')['day'].apply(day).reset_index().rename({0:'day1'},axis=1)
i.drop(['ship','level_1'],axis=1,inplace=True)
ti=pd.concat([t,i],axis=1)
ti.drop('day',axis=1,inplace=True)
t1=pd.pivot(ti,index='ship',columns='day1',values='dx').reset_index().rename({1:'day1_x',2:'day2_x',3:'day3_x',4:'day4_x'},axis=1)
t1.loc[t1['day1_x']==0.,'day1_x']=1e-8
t1.loc[t1['day2_x']==0.,'day2_x']=1e-8
t1.loc[t1['day3_x']==0.,'day3_x']=1e-8
t1.loc[t1['day4_x']==0.,'day4_x']=1e-8
t21=t1.fillna(-1)




#%%
# 距离
train['dis'] = np.sqrt(train['x']**2 + train['y']**2)
#%%
test['dis'] = np.sqrt(test['x']**2 + test['y']**2)




#%%
# 聚类
def cluster(train,test):
    train['data_type'] = 0
    test['data_type'] = 1
    data = pd.concat([train, test], axis=0, join='outer')

    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
    data['cluster']= pd.DataFrame(gmm.fit_predict(data[['x','y','v','d']]))

    return data[['data_type','cluster']]
clu=cluster(train,test)
#%%
def rat(col):
    rat=col.value_counts(1)
    a=np.zeros((1,3))
    j=0
    for i in rat.index:
        a[0,i]=rat.values[j]
        j+=1
    a=pd.DataFrame(a,columns=['r0','r1','r2'])
    return a
t=test1.groupby('ship')['cluster'].apply(rat).reset_index().drop(['level_1','ship'],axis=1)
#%%
# 众数
def mode(col):
    a=col.mode()
    return a[0]




#%%
# PCA降维
def tran(col):
    # print(type(col))
    a = pd.DataFrame(col.values.T)
    # time.sleep(10)
    return a
pca = PCA(n_components=10)
#%%
x=train.groupby('ship')[['x']].apply(tran)
x=x.fillna(0)
x = pd.DataFrame(pca.fit_transform(x))
x=x.reset_index().rename({'index':'ship'},axis=1)
y=train.groupby('ship')[['y']].apply(tran)
y=y.fillna(0)
y = pd.DataFrame(pca.fit_transform(y))
y=y.reset_index().rename({'index':'ship'},axis=1)
#%%
x=test.groupby('ship')[['x']].apply(tran)
x=x.fillna(0)
x = pd.DataFrame(pca.fit_transform(x))
x=x.reset_index().rename({'index':'ship'},axis=1)
y=test.groupby('ship')[['y']].apply(tran)
y=y.fillna(0)
y = pd.DataFrame(pca.fit_transform(y))
y=y.reset_index().rename({'index':'ship'},axis=1)




#%%
# 特征生成
train_label = train.drop_duplicates('ship')

type_map = dict(zip(train_label['type'].unique(), np.arange(3)))
type_map_rev = {v:k for k,v in type_map.items()}
train_label['type'] = train_label['type'].map(type_map)

# 分位数
def quantile(col):
    a=col.quantile(0.75)
    return a

# 众数
def mode(col):
    a=col.mode()
    return a[0]

#聚合特征
def group_feature(df, key, target, aggs):  
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    # print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t

def extract_feature(df, train,choice=2):

    t = group_feature(df, 'ship','x',['count'])
    train = pd.merge(train, t, on='ship', how='left')

    train = pd.merge(train, t11, on='ship', how='left')
    # train = pd.merge(train, t12, on='ship', how='left')

    t=df.groupby('ship')['x'].apply(quantile).reset_index(name='x_quantile')
    train = pd.merge(train, t, on='ship', how='left',)
    t=df.groupby('ship')['y'].apply(quantile).reset_index(name='y_quantile')
    train = pd.merge(train, t, on='ship', how='left',)
    t=df.groupby('ship')['v'].apply(quantile).reset_index(name='v_quantile')
    train = pd.merge(train, t, on='ship', how='left',)
    t=df.groupby('ship')['d'].apply(quantile).reset_index(name='d_quantile')
    train = pd.merge(train, t, on='ship', how='left',)
    t=df.groupby('ship')['dis'].apply(quantile).reset_index(name='dis_quantile')
    train = pd.merge(train, t, on='ship', how='left',)

    t=df.groupby('ship')['x'].apply(mode).reset_index(name='x_mode')
    train = pd.merge(train, t, on='ship', how='left',)
    t=df.groupby('ship')['y'].apply(mode).reset_index(name='y_mode')
    train = pd.merge(train, t, on='ship', how='left',)
    t=df.groupby('ship')['v'].apply(mode).reset_index(name='v_mode')
    train = pd.merge(train, t, on='ship', how='left',)
    t=df.groupby('ship')['d'].apply(mode).reset_index(name='d_mode')
    train = pd.merge(train, t, on='ship', how='left',)
    t=df.groupby('ship')['dis'].apply(mode).reset_index(name='dis_mode')
    train = pd.merge(train, t, on='ship', how='left',)
    
    t = group_feature(df, 'ship','x',['max','min','mean','std','skew','sum','median'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','y',['max','min','mean','std','skew','sum','median'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','v',['max','min','mean','std','skew','sum','median'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','d',['max','min','mean','std','skew','sum','median'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','dis',['max','min','mean','std','skew','sum','median'])
    train = pd.merge(train, t, on='ship', how='left')

    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['y_max_y_min'] = train['y_max'] - train['y_min']
    train['y_max_x_min'] = train['y_max'] - train['x_min']
    train['x_max_y_min'] = train['x_max'] - train['y_min']
    train['dis_max_dis_min'] = train['dis_max'] - train['dis_min']

    train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min']==0, 0.001, train['x_max_x_min'])
    train['area'] = train['x_max_x_min'] * train['y_max_y_min']
    
    mode_hour = df.groupby('ship')['hour'].agg(lambda x:x.value_counts().index[0]).to_dict()
    train['mode_hour'] = train['ship'].map(mode_hour)
    
    hour_nunique = df.groupby('ship')['hour'].nunique().to_dict()
    date_nunique = df.groupby('ship')['date'].nunique().to_dict()
    train['hour_nunique'] = train['ship'].map(hour_nunique)
    train['date_nunique'] = train['ship'].map(date_nunique)

    t = df.groupby('ship')['time'].agg({'diff_time':lambda x:np.max(x)-np.min(x)}).reset_index()
    t['diff_day'] = t['diff_time'].dt.days
    t['diff_second'] = t['diff_time'].dt.seconds
    # second=t['diff_time'].dt.total_seconds()
    train = pd.merge(train, t, on='ship', how='left')

    return train

train_label = extract_feature(train, train_label)
#%%
# 训练
features = [x for x in train_label.columns if x not in ['fill','ship','type','time','diff_time','date','month', 'day','hour', 'minute']]

target = 'type'


#%%
# 调参
N_HYPEROPT_PROBES = 50
K_spilt=5
HYPEROPT_ALGO = tpe.suggest                 #  tpe.suggest OR hyperopt.rand.suggest

def get_lgb_params(space):
    lgb_params = dict()
    lgb_params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
    lgb_params['objective'] = 'multiclass'
    lgb_params['learning_rate'] = space['learning_rate']
    lgb_params['num_class'] = 3
    lgb_params['nthread'] = -1
    return lgb_params

obj_call_count = 0
cur_best_score = 0 # 0 or np.inf
log_writer = open( './lgb-hyperopt-log.txt', 'a+' )
be_lgb_pa={}

def objective(space):
    global obj_call_count, cur_best_score,be_lgb_pa

    obj_call_count += 1
    score=0

    print('LightGBM objective call #{} cur_best_score={:7.5f} best_param={}'.format(obj_call_count,cur_best_score,be_lgb_pa) )

    lgb_params = get_lgb_params(space)

    sorted_params = sorted(space.items(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('\nParams: {}'.format(params_str) )
    
    kf = StratifiedKFold(n_splits=K_spilt, shuffle=True, random_state=42)
    X = train_label[features].copy()
    y = train_label[target]
    out_of_fold = np.zeros((len(X), 3))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X,y)):
        D_train = lgb.Dataset(X.iloc[train_idx], label=y[train_idx])
        D_val = lgb.Dataset(X.iloc[val_idx], label=y[val_idx])
        # Train
        num_round = 10000
        clf = lgb.train(lgb_params,
                           D_train,
                           num_boost_round=num_round,
                           valid_sets=D_val,
                           # feval=None,
                           early_stopping_rounds=100,
                           verbose_eval=False,
                           )
        # predict
        nb_trees = clf.best_iteration
        val_pred = clf.predict(X.iloc[val_idx], num_iteration=nb_trees)
        out_of_fold[val_idx] = val_pred
        val_y = y[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        print('{} val f1: {}'.format(fold , metrics.f1_score(val_y, val_pred, average='macro')))

    oof = np.argmax(out_of_fold, axis=1)
    score= metrics.f1_score(y,oof, average='macro')
    print('oof f1={}'.format(score))


    log_writer.write('score={} Params:{} nb_trees={}\n'.format(score, params_str, nb_trees ))
    log_writer.flush()

    if score>cur_best_score:
        cur_best_score = score
        be_lgb_pa=lgb_params
        print('NEW BEST SCORE={}'.format(cur_best_score))
    return {'loss': -score, 'status': STATUS_OK}

space ={
        'learning_rate': hp.uniform('learning_rate', 0.009, 0.31),
       }

trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

print('The best params:')
print( be_lgb_pa )
print('\n\n')
log_writer.write('The best params:{}\n'.format(be_lgb_pa))
log_writer.flush()




#%%
# 训练结果
def lgb_model2():
    train_label = train.drop_duplicates('ship')

    train_label['type'].value_counts(1)

    type_map = dict(zip(train_label['type'].unique(), np.arange(3)))
    type_map_rev = {v:k for k,v in type_map.items()}
    train_label['type'] = train_label['type'].map(type_map)

    # 分位数
    def quantile(col):
        a=col.quantile(0.75)
        return a

    # 众数
    def mode(col):
        a=col.mode()
        return a[0]

    #聚合特征
    def group_feature(df, key, target, aggs):  
        agg_dict = {}
        for ag in aggs:
            agg_dict[f'{target}_{ag}'] = ag
        # print(agg_dict)
        t = df.groupby(key)[target].agg(agg_dict).reset_index()
        return t

    def extract_feature(df, train,choice=2):

        t = group_feature(df, 'ship','x',['count'])
        train = pd.merge(train, t, on='ship', how='left')

        t=df.groupby('ship')['x'].apply(quantile).reset_index(name='x_quantile')
        train = pd.merge(train, t, on='ship', how='left',)
        t=df.groupby('ship')['y'].apply(quantile).reset_index(name='y_quantile')
        train = pd.merge(train, t, on='ship', how='left',)
        t=df.groupby('ship')['v'].apply(quantile).reset_index(name='v_quantile')
        train = pd.merge(train, t, on='ship', how='left',)
        t=df.groupby('ship')['d'].apply(quantile).reset_index(name='d_quantile')
        train = pd.merge(train, t, on='ship', how='left',)

        t=df.groupby('ship')['x'].apply(mode).reset_index(name='x_mode')
        train = pd.merge(train, t, on='ship', how='left',)
        t=df.groupby('ship')['y'].apply(mode).reset_index(name='y_mode')
        train = pd.merge(train, t, on='ship', how='left',)
        t=df.groupby('ship')['v'].apply(mode).reset_index(name='v_mode')
        train = pd.merge(train, t, on='ship', how='left',)
        t=df.groupby('ship')['d'].apply(mode).reset_index(name='d_mode')
        train = pd.merge(train, t, on='ship', how='left',)

        t = group_feature(df, 'ship','x',['max','min','mean','std','skew','sum','median'])
        train = pd.merge(train, t, on='ship', how='left')
        t = group_feature(df, 'ship','y',['max','min','mean','std','skew','sum','median'])
        train = pd.merge(train, t, on='ship', how='left')
        t = group_feature(df, 'ship','v',['max','min','mean','std','skew','sum','median'])
        train = pd.merge(train, t, on='ship', how='left')
        t = group_feature(df, 'ship','d',['max','min','mean','std','skew','sum','median'])
        train = pd.merge(train, t, on='ship', how='left')

        train['x_max_x_min'] = train['x_max'] - train['x_min']
        train['y_max_y_min'] = train['y_max'] - train['y_min']
        train['y_max_x_min'] = train['y_max'] - train['x_min']
        train['x_max_y_min'] = train['x_max'] - train['y_min']

        train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min']==0, 0.001, train['x_max_x_min'])
        train['area'] = train['x_max_x_min'] * train['y_max_y_min']
        
        mode_hour = df.groupby('ship')['hour'].agg(lambda x:x.value_counts().index[0]).to_dict()
        train['mode_hour'] = train['ship'].map(mode_hour)
        
        hour_nunique = df.groupby('ship')['hour'].nunique().to_dict()
        date_nunique = df.groupby('ship')['date'].nunique().to_dict()
        train['hour_nunique'] = train['ship'].map(hour_nunique)
        train['date_nunique'] = train['ship'].map(date_nunique)

        t = df.groupby('ship')['time'].agg({'diff_time':lambda x:np.max(x)-np.min(x)}).reset_index()
        t['diff_day'] = t['diff_time'].dt.days
        t['diff_second'] = t['diff_time'].dt.seconds
        train = pd.merge(train, t, on='ship', how='left')

        return train

    train_label = extract_feature(train, train_label,0)


    features = [x for x in train_label.columns if x not in ['fill','ship','type','time','diff_time','month','date', 'day','hour', 'minute']]
    target = 'type'

    params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'learning_rate': 0.07141800435870115,
            'num_leaves': 142,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.016306561549273645,
            'lambda_l2': 0.8515548606376679,
            'max_bin': 178,
            'feature_fraction': 0.6801860243655539,
            'bagging_fraction': 0.8848920441608762,
            'bagging_freq': 3,
            'max_depth': 7,
            'num_class': 3,
            'nthread': -1
    }

    #%%
    K_fold=5
    #%%
    fold = StratifiedKFold(n_splits=K_fold, shuffle=True, random_state=42)

    X = train_label[features].copy()
    y = train_label[target]
    models = []
    pred = np.zeros((len(test_label),3))
    oof = np.zeros((len(X), 3))
    for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):

        train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
        val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])

        model = lgb.train(params, train_set, valid_sets=[train_set, val_set], 
                            verbose_eval=100,
                            num_boost_round=10000,
                            # feval=None,
                            early_stopping_rounds=100,)
        models.append(model)
        val_pred = model.predict(X.iloc[val_idx])
        oof[val_idx] = val_pred
        val_y = y.iloc[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        print(index, 'val f1: ', metrics.f1_score(val_y, val_pred, average='macro'))

        test_pred = model.predict(test_label[features])
        pred += test_pred/K_fold

    oof = np.argmax(oof, axis=1)
    print('\noof f1：{}\n'.format(metrics.f1_score(y,oof, average='macro',),))

pred=lgb_model2()

#%%
# 模型融合
def boosting():
    fold = StratifiedKFold(n_splits=K_fold, shuffle=True, random_state=42)
    X = train_label[features].copy()
    y = train_label[target]
    X1 = test_label[features].copy()
    "===================================第一轮========================================================"
    # models = []
    pred_feat = np.zeros((len(test_label),3))
    oof_feat = np.zeros((len(X), 3))
    for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):

        train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
        val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])

        model = lgb.train(params, train_set, valid_sets=[train_set, val_set], 
                            verbose_eval=100,
                            num_boost_round=10000,
                            # feval=None,
                            early_stopping_rounds=100,)
        # models.append(model)
        val_pred = model.predict(X.iloc[val_idx],num_iteration=model.best_iteration)
        oof_feat[val_idx] = val_pred
        val_y = y.iloc[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        print(index, 'val f1: ', metrics.f1_score(val_y, val_pred, average='macro'))

        test_pred = model.predict(X1,num_iteration=model.best_iteration)
        pred_feat += test_pred/K_fold

    oof = np.argmax(oof_feat, axis=1)
    print('\noof f1：{}\n'.format(metrics.f1_score(y,oof, average='macro',),))

    tr1 =  pd.DataFrame(oof_feat,columns=['f1','f2','f3'])
    te1 = pd.DataFrame(test_pred,columns=['f1','f2','f3'])
    X = pd.concat([X,tr1],axis=1)
    X1 = pd.concat([X1,te1],axis=1)
    "===================================第二轮========================================================"
    # models = []
    pred_feat = np.zeros((len(test_label),3))
    oof_feat = np.zeros((len(X), 3))
    for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):

        train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
        val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])

        model = lgb.train(params, train_set, valid_sets=[train_set, val_set], 
                            verbose_eval=100,
                            num_boost_round=10000,
                            # feval=None,
                            early_stopping_rounds=100,)
        # models.append(model)
        val_pred = model.predict(X.iloc[val_idx],num_iteration=model.best_iteration)
        oof_feat[val_idx] = val_pred
        val_y = y.iloc[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        print(index, 'val f1: ', metrics.f1_score(val_y, val_pred, average='macro'))

        test_pred = model.predict(X1,num_iteration=model.best_iteration)
        pred_feat += test_pred/K_fold

    oof = np.argmax(oof_feat, axis=1)
    print('\noof f1：{}\n'.format(metrics.f1_score(y,oof, average='macro',),))

    tr1 =  pd.DataFrame(oof_feat,columns=['f4','f5','f6'])
    te1 = pd.DataFrame(test_pred,columns=['f4','f5','f6'])
    X = pd.concat([X,tr1],axis=1)
    X1 = pd.concat([X1,te1],axis=1)
    "=======================第三轮========================================================"
    # models = []
    pred_feat = np.zeros((len(test_label),3))
    oof_feat = np.zeros((len(X), 3))
    for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):

        train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
        val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])

        model = lgb.train(params, train_set, valid_sets=[train_set, val_set], 
                            verbose_eval=100,
                            num_boost_round=10000,
                            # feval=None,
                            early_stopping_rounds=100,)
        # models.append(model)
        val_pred = model.predict(X.iloc[val_idx],num_iteration=model.best_iteration)
        oof_feat[val_idx] = val_pred
        val_y = y.iloc[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        print(index, 'val f1: ', metrics.f1_score(val_y, val_pred, average='macro'))

        test_pred = model.predict(X1,num_iteration=model.best_iteration)
        pred_feat += test_pred/K_fold

    oof = np.argmax(oof_feat, axis=1)
    print('\noof f1：{}\n'.format(metrics.f1_score(y,oof, average='macro',),))
    
    return pred_feat
    
pred=boosting()

#%%
# 输出结果
pred_cla = np.argmax(pred, axis=1)
sub = test_label[['ship']]
sub['pred'] = pred_cla
print(sub['pred'].value_counts(1))
sub['pred'] = sub['pred'].map(type_map_rev)
#%%
sub.to_csv('xv_dis_cal_10.csv', index=None, header=None)





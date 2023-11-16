#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv('./testset.csv')

df = df[['datetime_utc', ' _tempm',' _pressurem',' _hum',' _dewptm',' _conds']]
df.columns = ['date','tempm','pressurem','hum','dewptm','conds']

df['date'] = pd.to_datetime(df['date'], format='%Y%m%d-%H:%M')
df['date'] = df['date'].apply(lambda x: x.strftime('%Y/%m/%d %H:%M:%S'))


df = df.set_index('date')['2010':'2017']

df = df.sort_index()

df['conds'].unique()

# 将空值填充为特定的值（例如-1）
df['conds'].fillna(method='ffill')

# 将文本型离散变量转换为整数型离散变量
df['conds'] = pd.factorize(df['conds'])[0]

df.conds.unique()

for var in df.columns:
    df[var].fillna(df[var].rolling(6, min_periods=1).mean(), inplace=True)

# df.reset_index().to_csv('new_delhi_2010_2016.csv',index=False)

df = df.reset_index()
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour

# 定义要筛选的小时列表
hours_to_keep = [0, 3, 6, 9, 12, 15, 18, 21]
# 根据小时列表进行筛选
df = df[df['hour'].isin(hours_to_keep)]
df['minute'] = df['date'].dt.minute

# 筛选出分钟部分为 0 的数据
df = df[df['minute'] == 0]

df.drop(['hour', 'minute'], axis=1, inplace=True)
df.reset_index().to_csv('./new_delhi_2010_2016.csv',index=False)


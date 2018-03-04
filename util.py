from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.classifier import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression,Lasso,Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,roc_auc_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,LabelEncoder,Imputer
from sklearn.feature_selection import VarianceThreshold
#import matplotlib.pyplot as plt
import copy
import numpy as np
from collections import Counter
import warnings
file_path = '/data/examples/trend/data/'
rawdata_path = file_path + 'query_log/'

############ etl ############

def clean_df(df):
    df['ProductID'] = df['ProductID'].astype('str')
    df = df.replace(['055649'],['55649'])
    return df

#FileID被各個ProductID開啟的次數的比例
def get_file_product_count_percentage(df,df_perc=None,normalize=False):
    dft = df[['FileID','ProductID']]
    dft = dft.assign(Count=1)
    dft = dft.groupby(['FileID','ProductID'],as_index = False).sum().pivot('FileID','ProductID').fillna(0)
    if normalize:
        dft = dft.div(dft.sum(axis=1), axis=0)
    cols = [col+'_count_percentage' for col in list(dft.columns.get_level_values(1))]
    dft.columns = cols
    if df_perc is not None:
        rows = set(df_perc.index) - set(dft.index)
        for row in rows:
            dft.ix[row] = 0
        rows = set(dft.index) - set(df_perc.index)
        for row in rows:
            df_perc.ix[row] = 0
        dft = dft.add(df_perc)
    return dft

#每次被開啟的間隔時間的mean/std
def get_open_time(df,max_timestamp=None):
    if max_timestamp:
        df = pd.concat([max_timestamp,df],axis=0)
    dft = df[['FileID','QueryTs']]
    dft = dft.sort_values(by=['QueryTs'])
    dft['QueryTsInterval'] = dft.groupby('FileID')['QueryTs'].transform(pd.Series.diff)
    dft = dft.dropna()
    return dft, max_timestamp

def get_open_time_mean(df):
    dft = df.groupby('FileID')['QueryTsInterval'].mean()
    dft.name = 'QueryTsIntervalMean'
    return dft

def get_open_time_std(df):
    dft = df.groupby('FileID')['QueryTsInterval'].std()
    dft.name = 'QueryTsIntervalStd'
    return dft

############ etl ############

############ model ############
def get_data(version=4):
    cols = ['FileID','y']
    df = pd.read_csv('trend_v%s.csv'%version)
    df = df.set_index('FileID')
    test = pd.read_csv(file_path+'testing-set.csv',header=None)
    train = pd.read_csv(file_path+'training-set.csv',header=None)
    test.columns = cols
    train.columns = cols
    train = train.set_index('FileID')
    test = test.set_index('FileID')
    train_indices = train.index
    test_indices = test.index
    train = pd.concat([df.ix[train_indices],train],axis=1)
    y = train.pop('y')
    test = df.ix[test_indices]
    return train, y, test

############ model ############


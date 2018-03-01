import pandas as pd

#FileID被各個ProductID開啟的次數的比例
def get_file_product_count_percentage(df,normalize=False):
    dft = df[['FileID','ProductID']]
    dft = dft.assign(Count=1)
    dft = dft.groupby(['FileID','ProductID'],as_index = False).sum().pivot('FileID','ProductID').fillna(0)
    if normalize:
        dft = dft.div(dft.sum(axis=1), axis=0)
    cols = [col+'_count_percentage' for col in list(dft.columns.get_level_values(1))]
    dft.columns = cols
    return dft

#每次被開啟的間隔時間的mean/std
def get_open_time(df,max_timestamp=None):
    if max_timestamp:
        df = pd.concat([max_timestamp,df],axis=0)
    dft = df[['FileID','QueryTs']]
    dft['QueryTsInterval'] = dft.groupby('FileID')['QueryTs'].transform(pd.Series.diff)
    dft = dft.dropna()
    return dft   

def get_open_time_mean(df):
    return df.groupby('FileID')['QueryTsInterval'].mean()

def get_open_time_std(df):
    return df.groupby('FileID')['QueryTsInterval'].std()

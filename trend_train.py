import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# # temp
# def load_data(path=PATH, load_file_name=LOAD_FILE_NAME):

# 	query_csv_path = os.path.join(path, load_file_name)
# 	query_dd = pd.read_csv(query_csv_path)

# 	train_csv_path = os.path.join(path, load_file_name)
# 	train_dd = pd.read_csv(train_csv_path)

# 	return pd.read_csv(csv_path)


# load data

train_csv_path = 'training-set.csv'
test_csv_path = 'testing-set.csv'
query_csv_path = 'query_log_temp/'

train_label = pd.read_csv(train_csv_path, header=None)
train_label.columns = ['FileID', 'Label']

test_label = pd.read_csv(test_csv_path, header=None)
test_label.columns = ['FileID', 'Label']



### create training data

all_files = glob.glob(os.path.join(query_csv_path, "*.csv"))  
query_temp = (pd.read_csv(f, header=None) for f in all_files)
query_dd = pd.concat(query_temp, axis=0)
query_dd.columns = ['FileID', 'CustomerID', 'QueryTs', 'ProductID']





# extract data
train_dd = pd.merge(train_label, query_dd,on="FileID")
test_dd = pd.merge(test_label,query_dd,on="FileID")



# Create Feature

train_

# FileID, 



# tr
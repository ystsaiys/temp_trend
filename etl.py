import glob
file_path = '/data/examples/trend/data/'
rawdata_path = file_path + 'query_log/'
files = glob.glob(rawdata_path+'/*.csv') 
list.sort(files)
for f in files:
    pass

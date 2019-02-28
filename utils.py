import s3fs
import scipy.stats
import time
import multiprocessing as mp
import matplotlib.pylab as plt

import pandas as pd
import numpy as np
import scipy
import matplotlib.pylab as plt
import seaborn as sns
import ast
import glob

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn

def collect_featured_data_from_s3(path, fs):
    top_dir = fs.ls(path)
    data = []
    
    for files in top_dir:
        df = None
        bottom_dir = fs.ls(files)
        df = combine_multiple_json(bottom_dir)
        df = get_accleration_timeseries(df)
        params = [df, 'unkown', np.arange(0,100,10)]
        example = featurize(params)
        data.append(example)
        
    return data # a list that has a featurized vector (dictioanry) for each folder in top_dir (each capture session)
    
def mp_collect_featured_data_from_s3(path, fs):
    top_dir = fs.ls(path)
    data = []
    data_params = []
    
    for files in top_dir:
        df = None
        bottom_dir = fs.ls(files)
        df = combine_multiple_json(bottom_dir)
        df = get_accleration_timeseries(df)
        params = (df, 'unkown', np.arange(0,100,10))
        data_params.append(params)
        
    then = time.time()
    pool = mp.Pool(processes=8)
    data = pool.map(featurize,data_params)
    print((time.time()-then)/60, "minutes")
        
    return data # a list that has a featurized vector (dictioanry) for each folder in top_dir (each capture session)


def combine_multiple_json(bottom_dir):
    
    df = pd.DataFrame([])
    
    for partial_json  in bottom_dir:
        x = fs.open(partial_json)
        try: # TODO fix this try/except
            temp_data = pd.read_json(x.read())
        except ValueError:
            continue
        x.close()
        df = df.append(temp_data, ignore_index = True)
        temp_data = None
    
    df = pd.io.json.json_normalize(df['motion'])
    df = df.iloc[:,0:3]
        
    return df # 3xn dataframe of acceleration data   
        
def get_accleration_timeseries(timeseries):
    
    timeseries = timeseries.apply((lambda x: x**2))
    timeseries = timeseries.sum(axis=1)
    timeseries = timeseries.apply(np.sqrt)
    
    return timeseries # 1xn Series 

def featurize(params):
    ts = params[0]
    label = params[1]
    bins = params[2]
    mean = np.mean(ts)
    median = np.median(ts)
    std = np.std(ts)
    length = len(ts)
    kurtosis = scipy.stats.kurtosis(ts)
    
    n,b,p = plt.hist(ts, bins=bins)
    n = np.array(n)/float(np.sum(n)) #normalize i.e. fraction of entries in each bin
    
    if median == 0: 
        features = {'mean_over_median': 0, #dimensionless            
                    'std_over_median': 0, #dimensionless            
                    'length': length,
                    'kurtosis': kurtosis, #already dimensionless by definition
                   }
        
    else: 
        features = {'mean_over_median': mean/median, #dimensionless            
            'std_over_median': std/median, #dimensionless            
            'length': length,
            'kurtosis': kurtosis, #already dimensionless by definition
           }
        
    for i, val in enumerate(n):
        features[f'binfrac_{i}'] = val
    
    features['label'] = label
    
    
    return features
       
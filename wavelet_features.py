import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as b
from scipy import signal
from tsfresh import extract_features
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,confusion_matrix, roc_auc_score
import itertools
import threading
import math
import pyeeg
from scipy import signal
from pyquaternion import Quaternion
from scipy import stats
from scipy.fftpack import fft, fftfreq
import pywt
from collections import Counter
import scipy
from multiprocessing import Process, Value, Lock
from multiprocessing import Manager
from collections import Counter

df_reorient1 = pd.read_csv('clean_reorient_ori_1.csv')
df_reorient2 = pd.read_csv('clean_reorient_ori_2.csv')
df_reorient3 = pd.read_csv('clean_reorient_ori_3.csv')
df_reorient4 = pd.read_csv('clean_reorient_ori_4.csv')
df_reorient = pd.concat([df_reorient1, df_reorient2, df_reorient3, df_reorient4], ignore_index=True)
df_reorient.drop(['Unnamed: 0'], axis=1, inplace=True)
df_reorient['bookingID'] = df_reorient['bookingID'].astype(str)
# clean those with -1
df_reorient.drop(df_reorient[df_reorient['r_acc_mag'] == -1].index, inplace=True)
# clean accuracy zscore greater than 3
df_reorient = df_reorient[(np.abs(stats.zscore(df_reorient['Accuracy'])) < 3)]
# clean acc_magnitude
df_reorient = df_reorient[(np.abs(stats.zscore(df_reorient['r_acc_mag'])) < 3)]

# drop booking ID which has less than 30 readings
count_booking = df_reorient.groupby('bookingID').Speed.count()
id_to_be_dropped = list(count_booking[count_booking < 30].index)
df_reorient = df_reorient.drop(df_reorient[df_reorient['bookingID'].isin(id_to_be_dropped)].index, axis=0)

# group dataframe by bookingId
grouped = df_reorient.groupby('bookingID')
groups = dict(list(grouped))


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy
 
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    minima = np.nanmin(list_values)
    maxima = np.nanmax(list_values)
    minmax_diff = maxima - minima
    par_positive = maxima / mean
    if np.isnan(par_positive):
        par_positive = 0
    par_negative = minima / mean
    if np.isnan(par_negative):
        par_negative = 0
    kurtosis = stats.kurtosis(list_values)
    skew = stats.skew(list_values)
    sem = stats.sem(list_values)
    return [n5, n25, n75, n95, median, mean, std, var, rms, minima, maxima, minmax_diff, par_positive, par_negative,
           kurtosis, skew, sem]
 
def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) >= 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) <= np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def calculate_hjorth(list_values):
    mobility, complexity = pyeeg.hjorth(list_values)
    if np.isnan(mobility):
        mobility = 0
    if np.isnan(complexity):
        complexity = 0
    return [mobility, complexity]

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    hjorth = calculate_hjorth(list_values)
    return [entropy] + crossings + statistics + hjorth

def calculate_sma(item, second):
    duration = np.max(second) - np.mean(second)
    sum_ = 0
    for i in range(1, len(item)):
        current_item = item[i]
        prev_item = item[i-1]
        time_dif = second[i] - second[i-1]
        sum_ += (np.abs(current_item) + np.abs(prev_item)) * time_dif
        
    sma = sum_ * (1/(2*duration))
    return [sma]

def calculate_svm(item, second):
    n = len(item)
    svm = np.sqrt(item.dot(item)) / n
    return [svm]
                  
def extractFeatures(group, thread_no):
    df_features = []
    labels = []
    i = 0
    for key, item in group.items():
        i += 1
        if i % 100 == 0:
            print(thread_no, i)
        item_feature_columns = ['acceleration_x','acceleration_y','acceleration_z','gyro_x','gyro_y','gyro_z','Speed',
        'r_acceleration_x','r_acceleration_y','r_acceleration_z','r_gyro_x','r_gyro_y','r_gyro_z','r_acc_mag']
        features = []
        for column in item_feature_columns:
            list_coeff = pywt.wavedec(item[column], 'db1', level=3)
            for coeff in list_coeff:
                features += get_features(coeff)
        
        df_features.append(features)
        labels.append(item['label'].values[0])
    
    df = pd.DataFrame.from_records(df_features)
    df_label = pd.Series(labels)
    df.to_csv('features_wavelet_' + str(thread_no) + '.csv')
    df_label.to_csv('labels_wavelet_' + str(thread_no) + '.csv')
    

s1 = int(len(groups) * 0.25)
s2 = int(len(groups) * 0.5)
s3 = int(len(groups) * 0.75)
s4 = int(len(groups) * 1)
groups_slice1 = dict(itertools.islice(groups.items(), 0, s1))
groups_slice2 = dict(itertools.islice(groups.items(), s1, s2))
groups_slice3 = dict(itertools.islice(groups.items(), s2, s3))
groups_slice4 = dict(itertools.islice(groups.items(), s3, s4))

procs = []

p = Process(target=extractFeatures, args=(groups_slice1, 1))
procs.append(p)

p2 = Process(target=extractFeatures, args=(groups_slice2, 2))
procs.append(p2)

p3 = Process(target=extractFeatures, args=(groups_slice3, 3))
procs.append(p3)

p4 = Process(target=extractFeatures, args=(groups_slice4, 4))
procs.append(p4)

for p in procs: p.start()
for p in procs: p.join()
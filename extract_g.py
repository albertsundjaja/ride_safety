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
from multiprocessing import Process, Value, Lock
from multiprocessing import Manager
from scipy import stats

df = pd.read_csv('/path/to/measurements.csv')
df_label = pd.read_csv('/path/to/labels.csv')

# clean duplicates, counts: 18
clean_label = df_label.drop_duplicates(subset=['bookingID'], keep=False)

# change booking id datatype
df['bookingID'] = df['bookingID'].astype(str)
clean_label['bookingID'] = clean_label['bookingID'].astype(str)

# merge into one df
df_merge = df.merge(clean_label, on='bookingID')

# drop booking ID which has less than 30 readings
count_booking = df_merge.groupby('bookingID').Speed.count()
id_to_be_dropped = list(count_booking[count_booking < 30].index)
df_merge = df_merge.drop(df_merge[df_merge['bookingID'].isin(id_to_be_dropped)].index, axis=0)

# clean outliers with zscore >3 for 'second' column
df_merge = df_merge[(np.abs(stats.zscore(df_merge['second'])) < 3)]
# clean again for second that is greater than 1 hour
df_merge = df_merge[df_merge['second'] < 3600].copy()

# group dataframe by bookingId
grouped = df_merge.groupby('bookingID')
groups = dict(list(grouped))

N = 2 # filter order
Fc = 0.5 # Nyquist Frequency  
b,a = signal.butter(N,Fc)

window_size = 60
overlap = 30

def slidingWindow(data):
    """ do a sliding window of 60s with 50% overlap, get the window with most data """
    max_second = data['second'].max()
    no_of_windows = math.ceil(max_second / overlap)
    
    data_count_window_max = 0
    window_max = pd.DataFrame()
    for i in range(0, no_of_windows):
        bot = 0 + i * overlap
        top = window_size + i * overlap
        current_window = data.query('%s <= second and second <= %s' % (bot,top))

        # if there is nothing in this window go to next
        if (current_window.empty):
            continue
        
        if len(current_window) > data_count_window_max:
            data_count_window_max = len(current_window)
            window_max = current_window
        
    return window_max

def getG(window_max):
    filtered_ax = signal.filtfilt(b,a, window_max['acceleration_x'], padlen=int(len(window_max)/2))
    filtered_ay = signal.filtfilt(b,a, window_max['acceleration_y'], padlen=int(len(window_max)/2))
    filtered_az = signal.filtfilt(b,a, window_max['acceleration_z'], padlen=int(len(window_max)/2))
    g = np.array([filtered_ax[int(len(window_max)/2)],filtered_ay[int(len(window_max)/2)],filtered_az[int(len(window_max)/2)]])
    return g

def createGravityAdjustmentDf(groups, thread_no):
    """ create a df for each booking ID containing the g vector"""
    # df_g = pd.DataFrame()
    df_g = pd.DataFrame()
    i = 0
    for idx, booking in groups.items():
        i += 1
        if i % 100 == 0:
            print(thread_no, i)
        window_max = slidingWindow(booking)
        g = getG(window_max)
        df_g = df_g.append({
            'bookingID':booking['bookingID'].values[0],
            'g_x':g[0],
            'g_y':g[1],
            'g_z':g[2]
        }, ignore_index=True)
    
    print("finished ", thread_no)
    print(df_g.head())
    df_g.to_csv('g_' + str(thread_no) + '.csv')


s1 = int(len(groups) * 0.25)
s2 = int(len(groups) * 0.5)
s3 = int(len(groups) * 0.75)
groups_slice1 = dict(itertools.islice(groups.items(), 0, s1))
groups_slice2 = dict(itertools.islice(groups.items(), s1, s2))
groups_slice3 = dict(itertools.islice(groups.items(), s2, s3))
groups_slice4 = dict(itertools.islice(groups.items(), s3, len(groups)))
print('s1', s1)
print('s2', s2)
print('s3', s3)
print('s4', len(groups))
print('slice1', len(groups_slice1))
print('slice2', len(groups_slice2))
print('slice3', len(groups_slice3))
print('slice3', len(groups_slice4))

procs = []

p = Process(target=createGravityAdjustmentDf, args=(groups_slice1, 1))
procs.append(p)

p2 = Process(target=createGravityAdjustmentDf, args=(groups_slice2, 2))
procs.append(p2)

p3 = Process(target=createGravityAdjustmentDf, args=(groups_slice3, 3))
procs.append(p3)

p4 = Process(target=createGravityAdjustmentDf, args=(groups_slice4, 4))
procs.append(p4)

for p in procs: p.start()
for p in procs: p.join()

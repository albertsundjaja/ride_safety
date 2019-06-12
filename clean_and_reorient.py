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
from multiprocessing import Process, Value, Lock
from multiprocessing import Manager
from scipy import stats


df = pd.read_csv('/path/to/measurements.csv')
df_label = pd.read_csv('/path/to/labels.csv')

def substractGravity(row, g):
    filtered_x = row['acceleration_x'] - g[0]
    filtered_y = row['acceleration_y'] - g[1]
    filtered_z = row['acceleration_z'] - g[2]
    returnData = np.array([filtered_x, filtered_y, filtered_z])
    return returnData

def getQuaternion(v):
    """ returns a quaternion to for reorientation according to v (gravity) """
    
    # get rotation axis
    # v * v'
    axis_vector = np.array([-9.81 * v[2], 0, 9.81 * v[0]])
    axis_mag = np.linalg.norm(axis_vector)
    rotation_axis = axis_vector/axis_mag
    
    # angle of rotation
    v_mag = np.linalg.norm(v)
    alpha = math.acos(v[1]/v_mag)
    
    return Quaternion(axis=rotation_axis, angle=alpha) 

def cleanGravityRotateAndGetMagnitude(row):
    g_data = df_g[df_g['bookingID'] == row['bookingID']]
    # if we cant find the bookingID, that means this bookingID will be dropped
    if g_data.empty:
        # return -1, so we can clean the bookingID
        return pd.Series([-1,-1,-1,-1,-1,-1,-1])

    g = np.empty((3,))
    g[0] = g_data['g_x'].values
    g[1] = g_data['g_y'].values
    g[2] = g_data['g_z'].values
    filtered_acc = substractGravity(row, g)
    quaternion = getQuaternion(g)
    rotated_acc = quaternion.rotate(filtered_acc)

    rotated_gyro = quaternion.rotate(np.array([row['gyro_x'],row['gyro_y'],row['gyro_z']]))

    acc_mag = np.sqrt(rotated_acc.dot(rotated_acc))
    returnData = pd.Series([rotated_acc[0], rotated_acc[1], rotated_acc[2], acc_mag, rotated_gyro[0], rotated_gyro[1], rotated_gyro[2]])
    return returnData

def calculateGMag(x):
    """ calculate magnitude of the G vector """
    g_vector = np.array([x['g_x'], x['g_y'], x['g_z']])
    return np.sqrt(g_vector.dot(g_vector))

def processDfCleanAndReorient(df, proc_no):
    df[['r_acceleration_x','r_acceleration_y','r_acceleration_z', 'r_acc_mag', 'r_gyro_x', 'r_gyro_y', 'r_gyro_z']] = df.apply(lambda x: cleanGravityRotateAndGetMagnitude(x), axis=1)
    print('finished proc no ', str(proc_no))
    print(df.head())
    df.to_csv('clean_reorient_ori_' + str(proc_no) + '.csv')

df_g1 = pd.read_csv('g_1.csv')
df_g2 = pd.read_csv('g_2.csv')
df_g3 = pd.read_csv('g_3.csv')
df_g4 = pd.read_csv('g_4.csv')

# combine extracted g data
df_g = pd.concat([df_g1,df_g2,df_g3,df_g4], ignore_index = True)
df_g.drop(['Unnamed: 0'], axis=1, inplace=True)

df_g['g_mag'] = df_g.apply(lambda x: calculateGMag(x), axis=1)

# remove g data for bookingID which has g_mag zscore greater than 3
df_g = df_g[(np.abs(stats.zscore(df_g['g_mag'])) < 3)]

# change bookingID data type to str
df_g['bookingID'] = df_g['bookingID'].astype(str)

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

df_merge2 = df_merge.copy()
# clean again for 'second' that is greater than 2 hour
df_merge2 = df_merge2[df_merge2['second'] < 7200]

df_merge2 = df_merge2.sort_values('bookingID')


s1 = int(len(df_merge2) * 0.25)
s2 = int(len(df_merge2) * 0.5)
s3 = int(len(df_merge2) * 0.75)
df_merge2_a = df_merge2.iloc[:s1, :]
df_merge2_b = df_merge2.iloc[s1:s2, :]
df_merge2_c = df_merge2.iloc[s2:s3, :]
df_merge2_d = df_merge2.iloc[s3:, :]

print('processing')

procs = []

p = Process(target=processDfCleanAndReorient, args=(df_merge2_a, 1))
procs.append(p)

p2 = Process(target=processDfCleanAndReorient, args=(df_merge2_b, 2))
procs.append(p2)

p3 = Process(target=processDfCleanAndReorient, args=(df_merge2_c, 3))
procs.append(p3)

p4 = Process(target=processDfCleanAndReorient, args=(df_merge2_d, 4))
procs.append(p4)

for p in procs: p.start()
for p in procs: p.join()


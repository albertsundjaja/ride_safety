import math
from scipy import signal
import pandas as pd
import numpy as np
from pyquaternion import Quaternion
import pywt
from collections import Counter
import scipy
from scipy import stats
import pyeeg

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

def create_gravity_adjustment_df(groups):
    """ create a df for each booking ID containing the g vector"""
    df_g = pd.DataFrame()
    print("Extracting data. Number of bookingID: ", str(len(groups)))
    i = 0
    for idx, booking in groups.items():
        i += 1
        if i % 100 == 0:
            print("processing ", i)
        # clean out data with readings above 3600 to save time
        booking_cleaned = booking[booking['second'] < 3600]
        # if empty, fill with zeros
        if booking_cleaned.empty:
            df_g = df_g.append({
                'bookingID':booking['bookingID'].values[0],
                'g_x':0,
                'g_y':0,
                'g_z':0
            }, ignore_index=True)
            continue

        window_max = slidingWindow(booking_cleaned)
        g = getG(window_max)
        df_g = df_g.append({
            'bookingID':booking['bookingID'].values[0],
            'g_x':g[0],
            'g_y':g[1],
            'g_z':g[2]
        }, ignore_index=True)
    
    print("finished")
    return df_g

def create_gravity_adjustment_df_multi(groups, thread_no):
    """ create a df for each booking ID containing the g vector"""
    df_g = pd.DataFrame()
    print("Extracting data. Thread", str(thread_no), "Number of bookingID: ", str(len(groups)))
    i = 0
    for idx, booking in groups.items():
        i += 1
        if i % 100 == 0:
            print("thread", str(thread_no), "processing ", str(i))
        # clean out data with readings above 3600 to save time
        booking_cleaned = booking[booking['second'] < 3600]
        # if empty, fill with zeros
        if booking_cleaned.empty:
            df_g = df_g.append({
                'bookingID':booking['bookingID'].values[0],
                'g_x':0,
                'g_y':0,
                'g_z':0
            }, ignore_index=True)
            continue

        window_max = slidingWindow(booking_cleaned)
        g = getG(window_max)
        df_g = df_g.append({
            'bookingID':booking['bookingID'].values[0],
            'g_x':g[0],
            'g_y':g[1],
            'g_z':g[2]
        }, ignore_index=True)
    
    print("thread", str(thread_no), "finished")
    df_g.to_csv('multi_extract_g_' + str(thread_no) + ".csv")

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

def cleanGravityRotateAndGetMagnitude(row, df_g):
    g_data = df_g[df_g['bookingID'] == row['bookingID']]
    # if we cant find the bookingID
    if g_data.empty:
        # return 0
        return pd.Series([0,0,0,0,0,0,0], index=['r_acceleration_x','r_acceleration_y','r_acceleration_z', 'r_acc_mag', 'r_gyro_x', 'r_gyro_y', 'r_gyro_z'])

    g = np.empty((3,))
    g[0] = g_data['g_x'].values
    g[1] = g_data['g_y'].values
    g[2] = g_data['g_z'].values
    filtered_acc = substractGravity(row, g)
    quaternion = getQuaternion(g)
    rotated_acc = quaternion.rotate(filtered_acc)

    rotated_gyro = quaternion.rotate(np.array([row['gyro_x'],row['gyro_y'],row['gyro_z']]))

    acc_mag = np.sqrt(rotated_acc.dot(rotated_acc))
    returnData = pd.Series([rotated_acc[0], rotated_acc[1], rotated_acc[2], acc_mag, rotated_gyro[0], rotated_gyro[1], rotated_gyro[2]],
                index=['r_acceleration_x','r_acceleration_y','r_acceleration_z', 'r_acc_mag', 'r_gyro_x', 'r_gyro_y', 'r_gyro_z'])
    return returnData

def calculateGMag(x):
    """ calculate magnitude of the G vector """
    g_vector = np.array([x['g_x'], x['g_y'], x['g_z']])
    return np.sqrt(g_vector.dot(g_vector))


def process_clean_and_reorient(df, df_g):
    print('cleaning and reorienting ', len(df))
    newDf = pd.DataFrame()
    i = 0
    for key, row in df.iterrows():
        i += 1
        if i % 100 == 0:
            print('processed ', i)
        reoriented_series = cleanGravityRotateAndGetMagnitude(row, df_g)
        row = row.append(reoriented_series)
        newDf = newDf.append(row, ignore_index=True)
    print('finish')
    return newDf

def process_clean_and_reorient_multi(df, df_g, thread_no):
    newDf = pd.DataFrame()
    i = 0
    for key, row in df.iterrows():
        i += 1
        if i % 100 == 0:
            print("thread", str(thread_no),'processed ', str(i))
        reoriented_series = cleanGravityRotateAndGetMagnitude(row, df_g)
        row = row.append(reoriented_series)
        newDf = newDf.append(row, ignore_index=True)
    print('finish')
    newDf.to_csv('multi_reorient_' + str(thread_no) + '.csv')

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
                  
def extract_features(group):
    df_features = []
    labels = []
    i = 0
    for key, item in group.items():
        i += 1
        if i % 100 == 0:
            print('processed ', i)
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
    return df, df_label

def extract_features_multi(group, thread_no):
    df_features = []
    labels = []
    i = 0
    for key, item in group.items():
        i += 1
        if i % 100 == 0:
            print('thread',str(thread_no),'processed ', str(i))
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
    print('thread',str(thread_no),'finished')
    df.to_csv('multi_features_' + str(thread_no) + '.csv')
    df_label.to_csv('multi_labels_' + str(thread_no) + '.csv')
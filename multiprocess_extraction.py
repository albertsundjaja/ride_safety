import pandas as pd
import numpy as np
import pickle
import preprocess_tools as tools
import itertools
from multiprocessing import Process
from sklearn.metrics import precision_score,recall_score,accuracy_score,confusion_matrix, roc_auc_score

# load data to be predicted
df_measurement = pd.read_csv('/path/to/measurements.csv')
df_label = pd.read_csv('/path/to/labels.csv')

# change data type of bookingID to str
df_measurement['bookingID'] = df_measurement['bookingID'].astype(str)
df_label['bookingID'] = df_label['bookingID'].astype(str)
# combine the label and the measurement
df_merge = df_measurement.merge(df_label, on='bookingID')

# group dataframe by bookingId
grouped = df_merge.groupby('bookingID')
groups = dict(list(grouped))

if __name__ == "__main__":
    # extract g
    print('extracting g')
    s1 = int(len(groups) * 0.25)
    s2 = int(len(groups) * 0.5)
    s3 = int(len(groups) * 0.75)
    groups_slice1 = dict(itertools.islice(groups.items(), 0, s1))
    groups_slice2 = dict(itertools.islice(groups.items(), s1, s2))
    groups_slice3 = dict(itertools.islice(groups.items(), s2, s3))
    groups_slice4 = dict(itertools.islice(groups.items(), s3, len(groups)))

    procs = []

    p = Process(target=tools.create_gravity_adjustment_df_multi, args=(groups_slice1, 1))
    procs.append(p)

    p2 = Process(target=tools.create_gravity_adjustment_df_multi, args=(groups_slice2, 2))
    procs.append(p2)

    p3 = Process(target=tools.create_gravity_adjustment_df_multi, args=(groups_slice3, 3))
    procs.append(p3)

    p4 = Process(target=tools.create_gravity_adjustment_df_multi, args=(groups_slice4, 4))
    procs.append(p4)

    for p in procs: p.start()
    for p in procs: p.join()

    df_g1 = pd.read_csv('multi_extract_g_1.csv')
    df_g2 = pd.read_csv('multi_extract_g_2.csv')
    df_g3 = pd.read_csv('multi_extract_g_3.csv')
    df_g4 = pd.read_csv('multi_extract_g_4.csv')
    df_g = pd.concat([df_g1,df_g2,df_g3,df_g4], ignore_index=True)
    df_g.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_g['bookingID'] = df_g['bookingID'].astype(str)

    # clean and reorient
    print('reorienting')
    s1 = int(len(df_merge) * 0.25)
    s2 = int(len(df_merge) * 0.5)
    s3 = int(len(df_merge) * 0.75)
    df_merge_a = df_merge.iloc[:s1, :]
    df_merge_b = df_merge.iloc[s1:s2, :]
    df_merge_c = df_merge.iloc[s2:s3, :]
    df_merge_d = df_merge.iloc[s3:, :]

    procs = []

    p = Process(target=tools.process_clean_and_reorient_multi, args=(df_merge_a, df_g, 1))
    procs.append(p)

    p2 = Process(target=tools.process_clean_and_reorient_multi, args=(df_merge_b, df_g, 2))
    procs.append(p2)

    p3 = Process(target=tools.process_clean_and_reorient_multi, args=(df_merge_c, df_g, 3))
    procs.append(p3)

    p4 = Process(target=tools.process_clean_and_reorient_multi, args=(df_merge_d, df_g, 4))
    procs.append(p4)

    for p in procs: p.start()
    for p in procs: p.join()

    # extract features
    print('extracting features')
    df1 = pd.read_csv('multi_reorient_1.csv')
    df2 = pd.read_csv('multi_reorient_2.csv')
    df3 = pd.read_csv('multi_reorient_3.csv')
    df4 = pd.read_csv('multi_reorient_4.csv')
    df_merge = pd.concat([df1,df2,df3,df4], ignore_index=True)
    df_merge['bookingID'] = df_merge['bookingID'].astype(str)

    grouped_feature = df_merge.groupby('bookingID')
    groups_feature = dict(list(grouped_feature))

    s1 = int(len(groups_feature) * 0.25)
    s2 = int(len(groups_feature) * 0.5)
    s3 = int(len(groups_feature) * 0.75)
    s4 = int(len(groups_feature) * 1)
    groups_slice1 = dict(itertools.islice(groups_feature.items(), 0, s1))
    groups_slice2 = dict(itertools.islice(groups_feature.items(), s1, s2))
    groups_slice3 = dict(itertools.islice(groups_feature.items(), s2, s3))
    groups_slice4 = dict(itertools.islice(groups_feature.items(), s3, s4))

    procs = []

    p = Process(target=tools.extract_features_multi, args=(groups_slice1, 1))
    procs.append(p)

    p2 = Process(target=tools.extract_features_multi, args=(groups_slice2, 2))
    procs.append(p2)

    p3 = Process(target=tools.extract_features_multi, args=(groups_slice3, 3))
    procs.append(p3)

    p4 = Process(target=tools.extract_features_multi, args=(groups_slice4, 4))
    procs.append(p4)

    for p in procs: p.start()
    for p in procs: p.join()

    print('Extraction Finished')
# Ride Safety Detection

## Overview

This is a project to solve a challenge provided by AI for SEA. The problem is classifying dangerous driving based on mobile phone's accelerometer, gyroscope and GPS.

## Dataset

The dataset consists of 20,018 bookingID (with 18 duplicates) with measurements from accelerometer, gyroscope and GPS.

The data is dirty, which is most likely caused by sensors' measurement error. The axes of the mobile phones are also inconsistent, since it depends on where the driver is placing their phones. Also, different mobile phones have different sensors performance.


### Cleaning and Preprocessing

#### Duplicates

There are 18 duplicates for the label provided. They will be **dropped**.

#### Filtering gravity

The accelerometer measures both gravity, driver's acceleration, vehicle vibration and other noises. As gravity is constant and significant, we will filter it out using Butterworth low-pass filter. Resulting G's with z-score greater than 3 will be **dropped**. 

The cleaning and duplicates and gravity filtering is done in `extract_g.py` It is done outside the jupyter notebook to take advantage of multiprocessing for faster calculation.

#### Reorientation

To account for different orientation of the mobile phones, it is important that we rotate the axes so that data measurements has the same axes. We can do the rotation by using the gravity and get its quaternion for rotation.

the process is done in `clean_and_reorient.py`

#### Outliers

Data with GPS's accuracy reading and acceleration magnitude with z-score greater than 3 will be **dropped**.

## Features

The signals that we get from the sensors reading can be thought of as time-series waves. Thus, we will do the wave extraction (band extraction) using **wavelet transform**, the resulting signals will be extracted for statistical features.

### Features Extracted from the Transformed Wave

```
5, 25, 75, 95 percentile
median
mean
standard deviation
variance
residual mean square (RMS)
maximum
minimum
maximum and minimum difference
Peak-to-Average ratio
kurtosis
standard error of mean
entropy
zero-crossings
mean-crossings
hjorth parameters (mobility and complexity)
```






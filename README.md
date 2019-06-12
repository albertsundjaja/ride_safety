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

Some bookingID has measurement up to 3 hours time, it seems that this does not add significant performance and can slow down the training process. Thus, measurements that is greater than 3600 seconds will be **dropped** during training.

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
skew
standard error of mean
entropy
zero-crossings
mean-crossings
hjorth parameters (mobility and complexity)
```

## Classifiers

The classifier used for the problem is a **Random Forest classifier** from the sklearn library.

## Instructions For Making Prediction

For making prediction using the pretrained model

### Step 1: Clone

Clone this repo

```
git clone https://github.com/albertsundjaja/ride_safety.git
```

go into the folder

```
cd ride_safety
```

### Step 2: Install dependencies if not yet installed

To open `*.ipynb`, **jupyter notebook** is needed, to simplify the installation, please use **Anaconda** python distribution

the preprocessing library `preprocess_tools.py` is provided in this repo.

make sure that all dependencies have been installed either by using `pip` or `conda`

the dependencies are as below

```
numpy==1.16.3
pandas==0.23.4
pyeeg==0.4.4
pyquaternion==0.9.5
PyWavelets==1.0.3
scikit-learn==0.20.3
scipy==1.2.1
```

to install **pyeeg**, please follow the instructions in https://github.com/forrestbao/pyeeg

and for **pyquaternion**, please follow the instruction in http://kieranwynn.github.io/pyquaternion/

for **pywavelets**, https://pywavelets.readthedocs.io/en/latest/install.html

### Step 3: Preprocessing

**NOTE** there are 2 methods to do the preprocessing of the features, one uses jupyter notebook and the other uses python scripts to take advantage of **multiprocess**. If the data to be predicted is huge, plase use multiprocess method to quicken the process. Scroll down to **Use multiprocess method** section

#### Jupyter Notebook method

open up jupyter notebook

```
jupyter notebook
```

and open the `prediction_template.ipynb`


change the path to the measurements and labels data in this cell

```
# load data to be predicted
df_measurement = pd.read_csv('/path/to/measurements.csv')
df_label = pd.read_csv('/path/to/labels.csv')
```

it is assumed that the `measurements.csv` and `labels.csv` above follow the same format as the one in the training data provided

Run all the cells **in order**

#### Use multiprocess method

To quicken the preprocessing, it is better to take advantage of **multiprocessing**, please follow below steps to use multiprocess

fill in the path to the data in `multiprocess_extraction.py` (it is assumed that the `measurements.csv` and `labels.csv` follow the same format as the one in the training data provided)

```
# load data to be predicted
df_measurement = pd.read_csv('/path/to/measurements.csv')
df_label = pd.read_csv('/path/to/labels.csv')
```

run the script

```
python multiprocess_extraction.py
```

run jupyter notebook 

```
jupyter notebook
```

and open `prediction_template_multi.ipynb`

then run all the cells

## Training the Model On Your Own

fill out the path to the required data files in `extract_g.py` and `clean_and_orient.py`

```
df = pd.read_csv('/path/to/measurements.csv')
df_label = pd.read_csv('/path/to/labels.csv')
```

if using **windows**, the code will need to be enclosed in a `if __name__ == "__main__":` in order for multiprocessing to work

**macOS** does not need to do any of the below modifications. Note that it has not been tested in **linux** based OS yet.

modify the last part of `extract_g.py` into

```
if __name__ == "__main__":
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
```

for `clean_and_orient.py`

```
if __name__ == "__main__":
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
```

and for `wavelet_features.py`

```
if __name__ == "__main__":
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
```



run the scripts in order:

```
python extract_g.py
python clean_and_reorient.py
python wavelet_features.py
```

run jupyter notebook and open `training.ipynb`, and follow and run the cells.


## References and Citations

> Forrest S. Bao, Xin Liu and Christina Zhang, "PyEEG: An Open Source Python Module for EEG/MEG Feature Extraction," Computational Intelligence and Neuroscience, March, 2011
 
> Tundo, Marco & Lemaire, Edward & Baddour, Natalie. (2013). Correcting Smartphone Orientation for Accelerometer-Based Analysis. MeMeA 2013 - IEEE International Symposium on Medical Measurements and Applications, Proceedings. 10.1109/MeMeA.2013.6549706. 

> Lu, D. N., Nguyen, D. N., Nguyen, T. H., & Nguyen, H. N. (2018). Vehicle Mode and Driving Activity Detection Based on Analyzing Sensor Data of Smartphones. Sensors (Basel, Switzerland), 18(4), 1036. doi:10.3390/s18041036





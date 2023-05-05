#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:10:53 2023
from https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
@author: nico
"""

import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)

dataset.head()

#%% Preprocessing The first preprocessing step is to divide the dataset into a 
#feature set and corresponding labels. The following script performs this task:

X = dataset.drop('Class', 1)
y = dataset['Class']

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% As mentioned earlier, PCA performs best with a normalized feature set. 
# We will perform standard scalar normalization to normalize our feature set. 
#To do this, execute the following code:

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Why we use fit_transform() on training data but transform() on the test data?
#https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe

#%% Applying PCA
from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

#%% rellenando datos faltantes

# from sklearn.impute import SimpleImputer
# imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# #imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])

# X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
# imp_mean.fit(X)
# print(imp_mean.transform(X))
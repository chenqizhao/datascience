#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 10:32:07 2018

@author: chenqi zhao
"""

# data read in and process

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

data = pd.read_csv('Skyserver_20000.csv')

catagory = []
for i in data['class']:
    if i == 'STAR':
        catagory.append(1.0)
    elif i == 'GALAXY':
        catagory.append(2.0)
    elif i == 'QSO':
        catagory.append(3.0)
data['catagory'] = catagory
data.drop(['class', 'objid', 'run', 'rerun', 'camcol', 'field', 'specobjid', 'redshift'], axis = 1, inplace = True)
X = data.iloc[:, 0:10].values
y = data.iloc[:, -1].values

# splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 33)

# feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# apply svm to classify the data
svc = SVC()
training_start = time.perf_counter()
svc.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = svc.predict(X_test)
prediction_end = time.perf_counter()
acc_svc = (preds == y_test).sum().astype(float) / len(preds)*100
svc_train_time = training_end-training_start
svc_prediction_time = prediction_end-prediction_start
print("svm accuracy: %3.2f" % (acc_svc))
print("Training time: %4.3f" % (svc_train_time * 1000))
print("Predict time: %6.3f" % (svc_prediction_time * 1000))
lgc_eval = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 10)
svc_kfold_avg = lgc_eval.mean()
print("svc 10 fold average: %6.5f" % (svc_kfold_avg))
cm = confusion_matrix(y_test, preds)
svc_basic_report = classification_report(y_test, preds)

# apply decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
training_start = time.perf_counter()
dtc.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = dtc.predict(X_test)
prediction_end = time.perf_counter()
acc_dtc = (preds == y_test).sum().astype(float) / len(preds)*100
dtc_train_time = training_end-training_start
dtc_prediction_time = prediction_end-prediction_start
print("decision tree accuracy: %3.2f" % (acc_dtc))
print("Training time: %4.3f ms" % (dtc_train_time * 1000))
print("Predict time: %6.5f ms" % (dtc_prediction_time * 1000))
dtc_eval = cross_val_score(estimator = dtc, X = X_train, y = y_train, cv = 10)
dtc_kfold_avg = dtc_eval.mean()
print("decision tree 10 fold average: %6.5f" % (nbc_kfold_avg))
cmd = confusion_matrix(y_test, preds)
dtc_basic_report = classification_report(y_test, preds)

# apply grid search to tune parameters for svc
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1.7],'kernel': ['linear']},
        {'C': [1.7],'kernel': ['rbf'], 'gamma':[0.5]}] #[0.1, 0.5, 0.01, 0.001, 0.9]}]
grid_search = GridSearchCV(estimator = svc,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy_2 = grid_search.best_score_
best_parameter_2 = grid_search.best_params_

# apply grid search to tune parameters for decision tree
from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth' : range(20,19999,10), 'min_samples_split' : range(2,100,2)}]
grid_search = GridSearchCV(estimator = dtc,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy_2 = grid_search.best_score_
best_parameter_2 = grid_search.best_params_

#re-do svc with optimized parameters
svc = SVC(C = 1.3, kernel= 'rbf', gamma = 0.5)
training_start = time.perf_counter()
svc.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = svc.predict(X_test)
prediction_end = time.perf_counter()
acc_svc = (preds == y_test).sum().astype(float) / len(preds)*100
svc_train_time = training_end-training_start
svc_prediction_time = prediction_end-prediction_start
print("svm accuracy with gridseach: %3.2f" % (acc_svc))
print("Training time: %4.3f ms: " % (svc_train_time * 1000))
print("Predict time: %6.5f ms: " % (svc_prediction_time * 1000))
svc_eval = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 10)
svc_kfold_avg = svc_eval.mean()
print("svc 10 fold average after tuning: %6.5f" % (svc_kfold_avg))

#re-do dtc with optimized parameters
dtc = DecisionTreeClassifier()
training_start = time.perf_counter()
dtc.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = dtc.predict(X_test)
prediction_end = time.perf_counter()
acc_dtc = (preds == y_test).sum().astype(float) / len(preds)*100
dtc_train_time = training_end-training_start
dtc_prediction_time = prediction_end-prediction_start
print("decision tree accuracy: %3.2f" % (acc_dtc))
print("Training time: %4.3f ms" % (dtc_train_time * 1000))
print("Predict time: %6.5f ms" % (dtc_prediction_time * 1000))
dtc_eval = cross_val_score(estimator = dtc, X = X_train, y = y_train, cv = 10)
dtc_kfold_avg = dtc_eval.mean()
print("decision tree 10 fold average: %6.5f" % (dtc_kfold_avg))
cmd = confusion_matrix(y_test, preds)
dtc_opt_report = classification_report(y_test, preds)

# apply PCA to extract the most impactful features
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_

# apply PCA to extract the most impactful features
pca = PCA(n_components = 9)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_

# redo svm after PCA
svc = SVC(kernel= 'linear')
training_start = time.perf_counter()
svc.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = svc.predict(X_test)
prediction_end = time.perf_counter()
acc_svc = (preds == y_test).sum().astype(float) / len(preds)*100
svc_train_time = training_end-training_start
svc_prediction_time = prediction_end-prediction_start
print("Scikit-Learn's Support Vector Machine Classifier's prediction accuracy is: %3.2f" % (acc_svc))
print("Time consumed for training: %4.3f ms" % (svc_train_time * 1000))
print("Time consumed for prediction: %6.5f ms" % (svc_prediction_time * 1000))
svc_eval = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 10)
svc_pca_kfold_avg = svc_eval.mean()
print("svc average with PCA = 4 is: %6.5f" % (svc_pca_kfold_avg))
cm = confusion_matrix(y_test, preds)
svc_final_report = classification_report(y_test, preds)


#redo decision tree after pca
dtc = DecisionTreeClassifier()
training_start = time.perf_counter()
dtc.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = dtc.predict(X_test)
prediction_end = time.perf_counter()
acc_dtc = (preds == y_test).sum().astype(float) / len(preds)*100
dtc_train_time = training_end-training_start
dtc_prediction_time = prediction_end-prediction_start
print("decision tree accuracy: %3.2f" % (acc_dtc))
print("Training time: %4.3f ms" % (dtc_train_time * 1000))
print("Predict time: %6.5f ms" % (dtc_prediction_time * 1000))
dtc_eval = cross_val_score(estimator = dtc, X = X_train, y = y_train, cv = 10)
dtc_kfold_avg = dtc_eval.mean()
print("decision tree 10 fold average: %6.5f" % (nbc_kfold_avg))
cmd = confusion_matrix(y_test, preds)
dtc_final_report = classification_report(y_test, preds)


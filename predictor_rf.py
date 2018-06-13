# -*- coding: utf-8 -*-
# @Time    : 2018/5/12 11:03
# @Author  : LeonHardt
# @File    : predictor_rf.py


import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data_path = os.getcwd()+"/data_imbalance/"
x_train = np.loadtxt(data_path+"x_adasyn_-1.txt", delimiter=',')
y_train = np.loadtxt(data_path+"y_adasyn_-1.txt", delimiter=',')
data_path = os.getcwd()+"/data/"
x_test = np.loadtxt(data_path+"x_test_a_-1.txt", delimiter=',')

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
# #
print("ready")
# im = Imputer(strategy="most_frequent")
# x_train = im.fit_transform(x_train)
# x_test = im.transform(x_test)
# lda = LinearDiscriminantAnalysis(n_components=50)
# x_train = lda.fit_transform(x_train, y_train)
# x_test = lda.transform(x_test)

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)
# prediction = gbm.predict(x_test)
# np.savetxt(os.getcwd()+"/prediction/prediction_filter_2.txt", prediction, delimiter=',')
prob = rf.predict_proba(x_test)
np.savetxt(os.getcwd()+"/prediction/rf1000_adasyn_-1.txt", prob, delimiter=',')
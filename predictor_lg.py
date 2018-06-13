# -*- coding: utf-8 -*-
# @Time    : 2018/5/15 17:42
# @Author  : LeonHardt
# @File    : predictor_lg.py

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

data_path = os.getcwd()+"/data/"

x_train = np.loadtxt(data_path+"x_train_most.txt", delimiter=',')
y_train = np.loadtxt(data_path+"y_train_filter.txt", delimiter=',')
x_test = np.loadtxt(data_path+"x_test_a_most.txt", delimiter=',')
print("ready")
# im = Imputer(strategy="most_frequent")
# x_train = im.fit_transform(x_train)
# x_test = im.transform(x_test)
lg = LogisticRegression(C=100, random_state=0)
ada = AdaBoostClassifier(base_estimator=lg, n_estimators=10, learning_rate=0.5)
ada.fit(x_train, y_train)
# prediction = gbm.predict(x_test)
# np.savetxt(os.getcwd()+"/prediction/prediction_filter_2.txt", prediction, delimiter=',')
prob = ada.predict_proba(x_test)
np.savetxt(os.getcwd()+"/prediction/ada_lg_most_2.txt", prob, delimiter=',')

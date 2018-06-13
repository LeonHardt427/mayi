# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 14:23
# @Author  : LeonHardt
# @File    : predictor_extrem.py

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import Imputer, StandardScaler

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
from score import score_roc
from sklearn.externals import joblib

data_path = os.getcwd()+"/data/"
# x_data = np.loadtxt(data_path+"x_train_train_most.txt", delimiter=',')
# y_data = np.loadtxt(data_path+"y_train_train.txt", delimiter=',')
x_cal = np.loadtxt(data_path+"x_train_cal_most.txt", delimiter=',')
y_cal = np.loadtxt(data_path+"y_train_cal.txt", delimiter=',')
x_test = np.loadtxt(data_path+"x_test_most.txt", delimiter=',')
# x_train_pca = np.loadtxt(os.getcwd()+'/PCA/x_train_most_pca.txt', delimiter=',')
# x_data = np.hstack((x_data, x_train_pca))
# x_test_pca = np.loadtxt(os.getcwd()+'/PCA/x_test_most_pca.txt', delimiter=',')
# x_test = np.hstack((x_test_org, x_test_pca))

# ext = joblib.load(filename="ext1000_pca.gz")

# print(ext.feature_importances_)
# selector = SelectFromModel(ext, prefit=True)
#
# x_train = selector.transform(x_data)
# x_test = selector.transform(x_test)

# rus = RandomUnderSampler(random_state=0)
# x_train_sample, y_data_sample = rus.fit_sample(x_train, y_data)
# print(x_train_sample.shape)
ex = ExtraTreesClassifier(n_estimators=1400, n_jobs=-1)
ex.fit(x_cal, y_cal)

prob = ex.predict_proba(x_test)
np.savetxt(os.getcwd()+"/prediction/ex1400_cal.txt", prob, delimiter=',')
# np.savetxt(os.getcwd()+"/prediction/ex1000_cw10_most.txt", prob, delimiter=',')
# -*- coding: utf-8 -*-
# @Time    : 2018/6/7 21:21
# @Author  : LeonHardt
# @File    : predictor_lgb_kaggle.py


import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler


# data_path = os.getcwd()+"/data/"
# train_file = pd.read_csv(data_path+"atec_anti_fraud_train.csv", delimiter=',', encoding='gbk', index_col=0)
#
# train_filter = train_file[train_file["label"] >= 0]
# y_label = train_filter["label"]
# train_filter.drop(["label", "date"], axis=1, inplace=True)
#
# test_df = pd.read_csv(data_path+"atec_anti_fraud_test_a.csv", delimiter=',', encoding='gbk', index_col=0)
# x_test = test_df.drop(['date'], axis=1).fillna(-1)
data_path = os.getcwd()+"/data/"
train_1 = np.loadtxt(data_path+"x_train_train_most.txt", delimiter=",")
train_2 = np.loadtxt(data_path+"x_train_cal_most.txt", delimiter=",")
train = np.vstack((train_1, train_2))
del train_1, train_2
y_1 = np.loadtxt(data_path+"y_train_train.txt", delimiter=",")
y_2 = np.loadtxt(data_path+"y_train_cal.txt", delimiter=",")
y_label = np.hstack((y_1, y_2))
del y_1, y_2

x_test = np.loadtxt(data_path+"x_test_most.txt", delimiter=",")

time = 6
opt_parameters = {'colsample_bytree': 0.8665631328558623,
                  'min_child_samples': 122,
                  'min_child_weight': 0.1,
                  'num_leaves': 48,
                  'reg_alpha': 2, 'reg_lambda': 50,
                  'subsample': 0.7252600946741159,
                  'scale_pos_weight': 2
                  }
clf_final = lgb.LGBMClassifier()
clf_final.set_params(n_estimators=4000, learning_rate=0.05, objective='binary', is_unbalance=True)
clf_final.set_params(**opt_parameters)
for i, ti in enumerate(range(time)):
    rus = RandomUnderSampler()
    x_train, y = rus.fit_sample(train, y_label)
    X_train, X_evl, y_train, y_evl= train_test_split(x_train,
                                                        y, test_size=0.10)

    fit_params = {"early_stopping_rounds":30,
                "eval_metric" : 'auc',
                "eval_set" : [(X_evl, y_evl)],
                'eval_names': ['valid'],
                'verbose': 100}
    clf_final.fit(X_train, y_train, **fit_params)
    if i == 0 :
        prob1 = clf_final.predict_proba(x_test)[:, 1]
    else:
        prob1 = np.hstack((prob1, clf_final.predict_proba(x_test)[:, 1]))
prob = np.mean(prob1, axis=0)
np.savetxt("lgb_kaggle4.txt", prob, delimiter=',')

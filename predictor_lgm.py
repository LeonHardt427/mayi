# -*- coding: utf-8 -*-
# @Time    : 2018/5/9 19:34
# @Author  : LeonHardt
# @File    : predictor_lgm.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from score import score_roc
from sklearn.externals import joblib

x_train = np.loadtxt(os.getcwd()+'/data_error/x_train_error93.txt', delimiter=',')
y_label = np.loadtxt(os.getcwd()+'/data_error/y_train_error93.txt', delimiter=',')
x_test = np.loadtxt(os.getcwd()+'/data_error/x_test_error93.txt', delimiter=',')
X_train, X_test, y_train, y_test = train_test_split(x_train, y_label, test_size=0.20, random_state=314)
# print(x_train_sample.shape)
gbm = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.05, objective='binary', is_unbalance=True,
                         colsample_bytree=0.8665631328558623,
                         min_child_samples=122, num_leaves=48, reg_alpha=2, reg_lambda=50,
                         subsample=0.7252600946741159, scale_pos_weight=2)

fit_params = {"early_stopping_rounds":30,
            "eval_metric" : 'auc',
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'verbose': 100}
gbm.fit(X_train, y_train, **fit_params)

prob = gbm.predict_proba(x_test)
np.savetxt(os.getcwd()+"/prediction/lgb4000_error_kaggle.txt", prob, delimiter=',')
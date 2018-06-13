# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 17:27
# @Author  : LeonHardt
# @File    : predictor_cp_lgm.py

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier

from nonconformist.base import ClassifierAdapter
from nonconformist.nc import ClassifierNc
from nonconformist.icp import IcpClassifier
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler
from nonconformist.acp import BootstrapConformalClassifier
from nonconformist.acp import CrossConformalClassifier

data_path = os.getcwd()+"/data/"

x_train = np.loadtxt(data_path+"x_train_filter.txt", delimiter=',')
y_train = np.loadtxt(data_path+"y_train_filter.txt", delimiter=',')
# x_train = np.loadtxt(data_path+"part_train.txt", delimiter=',')
# y_train = np.loadtxt(data_path+"part_label.txt", delimiter=',')
x_test = np.loadtxt(data_path+"x_test_a.txt")

gbm = lgb.LGBMClassifier(n_estimators=1000,)



# -----------------------------------------------------------------------------
# Define models
# -----------------------------------------------------------------------------
#
# models = {
# #           'ACP-RandomSubSampler'  : AggregatedCp(
# #                                         IcpClassifier(
# #                                             ClassifierNc(
# #                                                 ClassifierAdapter(DecisionTreeClassifier()))),
# #                                         RandomSubSampler()),
#             'ACP-CrossSampler'      : AggregatedCp(
#                                         IcpClassifier(
#                                             ClassifierNc(
#                                                 ClassifierAdapter(gbm))),
#                                         CrossSampler())
#           #   'ACP-BootstrapSampler'  : AggregatedCp(
#           #                               IcpClassifier(
#           #                                   ClassifierNc(
#           #                                       ClassifierAdapter(DecisionTreeClassifier()))),
#           #                               BootstrapSampler()),
#           #   'CCP'                   : CrossConformalClassifier(
#           #                               IcpClassifier(
#           #                                   ClassifierNc(
#           #                                       ClassifierAdapter(DecisionTreeClassifier())))),
#           #   'BCP'                   : BootstrapConformalClassifier(
#           #                               IcpClassifier(
#           #                                   ClassifierNc(
#           #                                       ClassifierAdapter(DecisionTreeClassifier()))))
#           }


model = AggregatedCp(
            IcpClassifier(
                ClassifierNc(
                    ClassifierAdapter(gbm))),
            CrossSampler())
model.fit(x_train, y_train)
print('predicting')
prediction = model.predict(x_test, significance=None)
np.savetxt(os.getcwd()+"/prediction/prediction_acp_cross_1.txt", prediction, delimiter=',')

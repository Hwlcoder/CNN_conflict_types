#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from textProcess3 import *
from sklearn.model_selection import StratifiedKFold
import os
import sklearn.naive_bayes as sk_bayes
from sklearn.externals import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#得到整个文本的矩阵（包括标签）
all_dir_1 = 'data/cate300_cut/'
all_dir_2 = 'data/parser/'
categories, cat_to_id = read_category()
all_xdata, all_ydata = process_file(all_dir_1,all_dir_2, cat_to_id)

#N折交叉验证，先将all_xdata和all_ydata分成N组
stratified_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=False)
count = 1
train_acc = []
test_acc = []

train_xdata = np.zeros((270,3,3))
train_labels = np.zeros(270)
test_xdata = np.zeros((30,3,3))
test_labels = np.zeros(30)
fp = open('result/'+'AccurancyResultNB.txt','w')
for train_index, test_index in stratified_folder.split(all_xdata, all_ydata):
    for i in range(len(train_index)):
        train_xdata[i] = all_xdata[train_index[i]]
        train_labels[i] = all_ydata[train_index[i]]
    for i in range(len(test_index)):
        test_xdata[i] = all_xdata[test_index[i]]
        test_labels[i] = all_ydata[test_index[i]]
    nsamples, nx, ny = train_xdata.shape
    d2_train_dataset = train_xdata.reshape((nsamples, nx * ny))
    nsamples1, nx1, ny1 = test_xdata.shape
    d2_test_dataset = test_xdata.reshape((nsamples1, nx1 * ny1))
    model = sk_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)  # 多项式分布的朴素贝叶斯
    model.fit(d2_train_dataset, train_labels)
    acc = model.score(d2_test_dataset, test_labels)  # 根据给定数据与标签返回正确率的均值
    print('朴素贝叶斯(高斯分布)模型评价: ',acc)
    test_acc.append(acc)
joblib.dump(model, "train_nb_model.m")
sum_acc = 0
for i in test_acc:
    sum_acc = sum_acc+i
print (sum_acc/10)

modeluse = joblib.load("train_nb_model.m")
my_data_path1 = "data/test/liu/test_3/"
my_data_path2 = "data/test/my/test_parser3/"
categories, cat_to_id = read_category()
test_1, test_2 = process_file(my_data_path1,my_data_path2,cat_to_id)
nsamples, nx, ny = test_1.shape
d2_test_data = test_1.reshape((nsamples, nx * ny))
print(test_2)
print(modeluse.predict(d2_test_data))
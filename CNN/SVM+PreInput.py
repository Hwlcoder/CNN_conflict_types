#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from textProcess import *
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.decomposition import PCA
from sklearn import svm
import time
from sklearn.externals import joblib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#得到整个文本的矩阵（包括标签）
all_dir = 'data/cate300_cut/'
categories, cat_to_id = read_category()
all_xdata, all_ydata = process_file(all_dir, cat_to_id)

#N折交叉验证，先将all_xdata和all_ydata分成N组
stratified_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=False)
train_acc = []
test_acc = []

train_xdata = np.zeros((270,29,29))
train_labels = np.zeros(270)
test_xdata = np.zeros((30,29,29))
test_labels = np.zeros(30)
fp = open('result/'+'AccurancyResultSVM.txt','w')
for train_index, test_index in stratified_folder.split(all_xdata, all_ydata):
    for i in range(len(train_index)):
        train_xdata[i] = all_xdata[train_index[i]]
        train_labels[i] = all_ydata[train_index[i]]
    for i in range(len(test_index)):
        test_xdata[i] = all_xdata[test_index[i]]
        test_labels[i] = all_ydata[test_index[i]]
    t = time.time()
    # svm方法
    nsamples, nx, ny = train_xdata.shape
    d2_train_dataset = train_xdata.reshape((nsamples, nx * ny))
    nsamples1, nx1, ny1 = test_xdata.shape
    d2_test_dataset = test_xdata.reshape((nsamples1, nx1 * ny1))
    pca = PCA(n_components=0.8, whiten=True)
    train_x = pca.fit_transform(d2_train_dataset)
    test_x = pca.transform(d2_test_dataset)
    svc = svm.SVC(kernel='rbf', C=10)
    svc.fit(train_x, train_labels)
    pre = svc.predict(test_x)
    acc = float((pre == test_labels).sum()) / len(test_x)
    test_acc.append(acc)
    print (u'准确率：%f,花费时间：%.2fs' % (acc, time.time() - t))
joblib.dump(svc, "train_svm_model.m")
print ("Done\n")

sum_acc = 0
for i in test_acc:
    sum_acc = sum_acc+i
print (sum_acc/10)

modeluse = joblib.load("train_svm_model.m")
my_data_path1 = "data/test/liu/test_3/"
categories, cat_to_id = read_category()
test_1, test_2 = process_file(my_data_path1,cat_to_id)
nsamples, nx, ny = test_1.shape
d2_test_data = test_1.reshape((nsamples, nx * ny))
d2_test_data_2 = pca.transform(d2_test_data)
print(test_2)
print(modeluse.predict(d2_test_data_2))

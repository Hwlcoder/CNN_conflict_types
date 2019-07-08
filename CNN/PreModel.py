#Alexnet
#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from textProcess import *
from sklearn.model_selection import StratifiedKFold
import matplotlib as plt
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#sess = tf.Session(config=tf.ConfigProto(device_count={'cpu':0}))

start = time.time()

#得到整个文本的矩阵（包括标签）
all_dir = 'data/cate300_cut/'
categories, cat_to_id = read_category()
all_xdata, all_ydata = process_file(all_dir, cat_to_id)

#N折交叉验证，先将all_xdata和all_ydata分成N组
stratified_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=False)
count = 1
train_cross_acc = []
test_cross_acc = []
wordsCount = 29
train_xdata = np.zeros((270, wordsCount, wordsCount))
train_labels = np.zeros(270)
test_xdata = np.zeros((30, wordsCount, wordsCount))
test_labels = np.zeros(30)
fp = open('result/'+'AccurancyResult.txt', 'w')

#定义网络模型
def my_conv_net(input_data):
#First Conv-relu-maxpool layer
    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')
# Second Conv-relu-maxpool layer
    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')
# Third Conv-relu-layer
    conv3 = tf.nn.conv2d(max_pool2, conv3_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_bias))
#第四层 Conv-relu-layer
    conv4 = tf.nn.conv2d(relu3, conv4_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_bias))
#第五层 Conv-relu-maxpool layer
    conv5 = tf.nn.conv2d(relu4, conv5_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_bias))
    max_pool3 = tf.nn.max_pool(relu5, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size3, max_pool_size3, 1], padding='SAME')
#将输出转换为一个[1xN],为下一个全连接层输入做准备
    final_conv_shape = max_pool3.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(max_pool3, [final_conv_shape[0], final_shape])
#First fully-connected layer
    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output,full1_weight), full1_bias))
# Second fully-connected layer
    fully_connected2 = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)
#第三层全连接层
    final_model_output = tf.add(tf.matmul(fully_connected2, full3_weight), full3_bias)
    return (final_model_output)


def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return (100. * num_correct / batch_predictions.shape[0])

#定义模型参数
batch_size = 256 #len(train_xdata)//2 #一个批量的图片数量
learning_rate = 0.02 #学习率
evaluation_size = len(test_xdata)  #模型验证数据集一个批量的数量
image_width = train_xdata[0].shape[0] 
image_height = train_xdata[0].shape[1] 
target_size = 3 #输出类别的个数
num_channels = 1 # 通道数为1
generations = 600 #迭代代数
eval_every = 5 #每次5个generation
conv1_features = 48 #卷积核的个数
conv2_features = 128 #卷积核的个数
conv3_features = 192  # 卷积核的个数
conv4_features = 192
conv5_features = 128
max_pool_size1 = 2 #池化层窗口大小
max_pool_size2 = 2 #池化层窗口大小
max_pool_size3 = 2  # 池化层窗口大小
fully_connected_size1 = 100 #全连接层大小

#定义训练权重和偏置的变量
#定义第一个卷积核的参数，其中用tf.truncated_normal生成正太分布的数据，#stddev（正态分布标准差）为0.1
conv1_weight = tf.get_variable('conv1',shape=[5, 5, num_channels, conv1_features], initializer=tf.contrib.layers.xavier_initializer())
#定义第一个卷积核对应的偏置
conv1_bias = tf.get_variable('c1', shape=[conv1_features], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
#定义第二个卷积核的参数，其中用tf.truncated_normal生成正太分布的数据，#stddev（正态分布标准差）为0.1
conv2_weight = tf.get_variable('conv2',shape=[3, 3, conv1_features, conv2_features], initializer=tf.contrib.layers.xavier_initializer())
#定义第二个卷积核对应的偏置
conv2_bias = tf.get_variable('c2', shape=[conv2_features], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
# 定义第三个卷积核的参数，其中用tf.truncated_normal生成正太分布的数据，#stddev（正态分布标准差）为0.1
conv3_weight = tf.get_variable('conv3',shape=[3, 3, conv2_features, conv3_features], initializer=tf.contrib.layers.xavier_initializer())
# 定义第三个卷积核对应的偏置
conv3_bias = tf.get_variable('c3', shape=[conv3_features], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
# 定义第4个卷积核的参数，其中用tf.truncated_normal生成正太分布的数据，#stddev（正态分布标准差）为0.1
conv4_weight = tf.get_variable('conv4',shape=[3, 3, conv3_features, conv4_features], initializer=tf.contrib.layers.xavier_initializer())
# 定义第4个卷积核对应的偏置
conv4_bias = tf.get_variable('c4', shape=[conv3_features], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
# 定义第5个卷积核的参数，其中用tf.truncated_normal生成正太分布的数据，#stddev（正态分布标准差）为0.1
conv5_weight = tf.get_variable('conv5',shape=[3, 3, conv4_features, conv5_features], initializer=tf.contrib.layers.xavier_initializer())
# 定义第5个卷积核对应的偏置
conv5_bias = tf.get_variable('c5', shape=[conv5_features], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

for train_index, test_index in stratified_folder.split(all_xdata, all_ydata):
    for i in range(len(train_index)):
        train_xdata[i] = all_xdata[train_index[i]]
        train_labels[i] = all_ydata[train_index[i]]
    for i in range(len(test_index)):
        test_xdata[i] = all_xdata[test_index[i]]
        test_labels[i] = all_ydata[test_index[i]]
    # print("Stratified Train Index:", train_index)
    # print("Stratified Test Index:", test_index)
    # print("Stratified y_train:", y[train_index])
    # print("Stratified y_test:", y[test_index],'\n')
    # train_xdata, train_labels = process_file(train_dir, cat_to_id)
    # test_xdata, test_labels = process_file(test_dir, cat_to_id)

    #定义数据集的占位符
    #输入数据的张量大小
    x_input_shape = (batch_size, image_width, image_height, num_channels)
    #创建输入训练数据的占位符
    x_input = tf.placeholder(tf.float32, shape=x_input_shape, name="x")
    #创建一个批量训练结果的占位符
    y_target = tf.placeholder(tf.int32, shape=batch_size, name="y_")
    #验证图片输入张量
    eval_input_shape = (evaluation_size, image_width, image_height,num_channels)
    #创建输入验证数据的占位符
    eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
    #创建一个批量验证结果的占位符
    eval_target = tf.placeholder(tf.int32, shape= evaluation_size )

    #定义全连接层的权重和偏置
    #输出卷积特征图的大小
    resulting_width = math.ceil(image_width / (max_pool_size1 * max_pool_size2 * max_pool_size3 ))
    resulting_height = math.ceil(image_height / (max_pool_size1 * max_pool_size2 * max_pool_size3))
    #将卷积层特征图拉成一维向量
    full1_input_size = resulting_width * resulting_height * conv5_features
    #创建第一个全连接层权重和偏置
    full1_weight =tf.Variable(tf.truncated_normal([full1_input_size,fully_connected_size1],stddev=0.1, dtype=tf.float32))
    full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1,dtype=tf.float32))
    #创建第二个全连接层权重和偏置
    full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1,fully_connected_size1],stddev=0.1, dtype=tf.float32))
    full2_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1,dtype=tf.float32))
    #创建第三个全连接层
    full3_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
    full3_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))


    #定义网络的训练数据和测数据
    model_output = my_conv_net(x_input)
    test_model_output = my_conv_net(eval_input)

    #使用Softmax函数作为loss function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))

    #创建一个训练和测试的函数
    prediction = tf.nn.softmax(model_output)
    test_prediction = tf.nn.softmax(test_model_output)

    # (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
    b = tf.constant(value=1, dtype=tf.float32)
    logits_eval = tf.multiply(prediction, b, name='logits_eval')

    # Create accuracy function
    #创建一个optimizer function
    my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_step = my_optimizer.minimize(loss)

    #开始训练模型
    # Initialize Variables
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    train_loss = []
    train_acc = []
    test_acc = []
    max_acc=0
    f=open('result/'+'%d.txt'%(count),'w')
    for i in range(generations):
        rand_index = np.random.choice(len(train_xdata), size=batch_size)
        rand_x = train_xdata[rand_index]
        rand_x = np.expand_dims(rand_x, 3)
        rand_y = train_labels[rand_index]
        # tf.add_to_collection("real_labels", rand_y)
        train_dict = {x_input: rand_x, y_target: rand_y}
        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss, temp_train_preds = sess.run([loss,prediction], feed_dict=train_dict)
        temp_train_acc = get_accuracy(temp_train_preds, rand_y)
        if (i+1) % eval_every == 0:
            eval_index = np.random.choice(len(test_xdata),size=evaluation_size)
            eval_x = test_xdata[eval_index]
            eval_x = np.expand_dims(eval_x, 3)
            eval_y = test_labels[eval_index]
            test_dict = {eval_input: eval_x, eval_target: eval_y}
            test_preds = sess.run(test_prediction, feed_dict=test_dict)
            temp_test_acc = get_accuracy(test_preds, eval_y)
    # Record and print results
            train_loss.append(temp_train_loss)
            train_acc.append(temp_train_acc)
            test_acc.append(temp_test_acc)
            acc_and_loss = [(i+1), temp_train_loss, temp_train_acc,temp_test_acc]
            acc_and_loss = [np.round(x,2) for x in acc_and_loss]
            f.write(str(i + 1) + ', val_acc: ' + str(temp_test_acc) + '\n')#保存模型
            if temp_test_acc > max_acc:
                max_acc = temp_test_acc
    train_cross_acc.append(acc_and_loss[2])
    test_cross_acc.append(acc_and_loss[3])
    saver.save(sess, './model/checkpoint/model.ckpt', global_step=i + 1)
    sess.close()
    f.close()

    fp.write('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
    fp.write('\n')
    print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
    count = count+1

#计算十折交叉的平均准确率
train_acc_sum = 0
test_acc_sum = 0
for i in train_cross_acc:
    train_acc_sum = train_acc_sum + i
train_acc_aver = train_acc_sum/10
for i in test_cross_acc:
    test_acc_sum = test_acc_sum + i
test_acc_aver = (test_acc_sum*1.0)/10
end = time.time()
timec = end - start

print('10 cross train acc is:%d'%train_acc_aver)
print('10 cross test acc is:%d'%test_acc_aver)
print('time:', timec)
fp.close()




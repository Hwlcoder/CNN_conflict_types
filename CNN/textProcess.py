# coding: utf-8
# encoding: utf-8
import sys
import importlib
importlib.reload(sys)
# import jieba.analyse  # 导入结巴jieba相关模块
import re
import numpy as np
from pylab import *  # 添加这行和mpl.rcParams可以解决汉字显示的问题
# get_ipython().magic('matplotlib inline')
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 汉字显示问题
import os
import tensorflow.contrib.keras as kr
wordsCount = 29
base_dir = 'data\\'
train_dir = os.path.join(base_dir, 'train\\')
test_dir = os.path.join(base_dir, 'test\\')
val_dir = os.path.join(base_dir, 'val\\')
# vocab_dir = os.path.join(base_dir, 'sougou.vocab.txt')

# CountResult = "D://Biyesheji/data/ConflictNormCSV/"
def textPreProcess(SourcePath):
    # 每两行的句子为一对构造矩阵
    # 1.统计每一行句子的单词数量，然后计算单词数量的分布，以便计算矩阵的长宽
    # 统计txt文件的行数
    file_raws = len(["" for line in open(SourcePath, "r", encoding='utf8')])
    print(file_raws)
    text = open(SourcePath, "r", encoding='utf8')
    couple = file_raws//2
    wordsAll = np.zeros((couple,wordsCount,wordsCount))  # 整个文档的矩阵
    label_1 = np.zeros(couple)  # 整个文档的句子对的标签（有多少句子对就有多少个标签）
    labels = label_1.astype(np.str)
    sen_length = []  # 统计每个句子的单词数量
    lines = []
    for line in text.readlines():
        words1 = re.split('[\(\)\[\]（）\"\'\,.;；、，。：\:“”【】&*《》<>%\n\u3000\ufeff\ue56e\u2022\ue4d2\ue148\ue3e5]', line)
        words2 = list(filter(lambda x: x != '' and x != ' ', words1))
        lines.append(words2)
        sen_length.append(len(words2))
    # print(sen_length)
    # print(lines)
    # wordsSum = 0
    # for i in sen_length:
    #     wordsSum = wordsSum + i
    # wordsAver = wordsSum // len(sen_length)  # 计算句子的平均单词数，作为矩阵的长和宽
    text.close()
    # 每相邻的句子之间构建矩阵
    csv_num = 1
    for raw_num in range(len(lines)-1):
        if raw_num % 2 == 1:
            continue
        else:
            com_num = wordsCount  # 确定两个句子比较的次数
            # a = [[]for i in range(wordsAver)]
            a = np.zeros(shape=(wordsCount, wordsCount))  # 初始化一个矩阵
            if len(lines[raw_num]) >= len(lines[raw_num + 1]):  # 比较两行的单词数量，多的作为列，少的作为行
                high = raw_num
                low = raw_num + 1
            else:
                high = raw_num + 1
                low = raw_num

            if len(lines[high]) <= wordsCount:
                com1 = len(lines[high])  # 长句子的比较次数
                com2 = len(lines[low])
            elif (len(lines[high]) >= wordsCount) & (len(lines[low]) <= wordsCount):
                com1 = wordsCount
                com2 = len(lines[low])
            else:
                com1 = wordsCount
                com2 = wordsCount
                # 构建一个wordsAver*wordsAver的矩阵
            #         print(com1,com2)
            for i in range(com1):
                for j in range(com2):
                    if (lines[high][i] == lines[low][j]):
                        a[j][i] = 1
        wordsAll[csv_num-1] = a
        csv_num = csv_num + 1
    label = os.path.basename(SourcePath)[:-8]
    for i in range(csv_num-1):
        labels[i] = label

    return wordsAll, labels
            # print(a)
            # dataframe = pd.DataFrame(a)
            # dataframe.to_csv(CountResult + '%d.csv' % (csv_num), index=False, sep=',', header=False)


def read_category():
    """读取分类目录，固定"""
    categories =['Action', 'Consequence', 'Subject']#'noConflict',
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def process_file(filepath, cat_to_id):
    """将文件转换为id表示"""
    action_content, action_category = textPreProcess(filepath + 'Action_cut.txt')
    Consequence_content, Consequence_category = textPreProcess(filepath + 'Consequence_cut.txt')
    #noconflict_content, noconflict_category = textPreProcess(filepath + 'noConflict_cut.txt')
    # Object_content, Object_category = textPreProcess(filepath + 'Object_cut.txt')
    Subject_content, Subject_category = textPreProcess(filepath + 'Subject_cut.txt')
    # contents = action_content
    contents = np.vstack((action_content, Consequence_content))
    #contents = np.vstack((contents, noconflict_content))
    # contents = np.vstack((contents, Object_content))
    contents = np.vstack((contents, Subject_content))
    labels = action_category
    labels = np.concatenate((action_category, Consequence_category), axis=0)
    #labels = np.concatenate((labels,noconflict_category), axis=0)
    # labels = np.concatenate((labels, Object_category), axis=0)
    labels = np.concatenate((labels, Subject_category), axis=0)
    #print(contents)
    label_id = []
    for i in range(len(labels)):
        label_id.append(cat_to_id[labels[i]])
    # print(label_id)
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    # x_pad = kr.preprocessing.sequence.pad_sequences(contents, max_length)
    # y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示
    x_pad = contents
    y_pad = np.array(label_id) # 将标签转换为one-hot表示
    return x_pad, y_pad


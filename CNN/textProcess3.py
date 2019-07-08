# coding: utf-8
# encoding: utf-8
import sys
import importlib
from  cilin import  CilinSimilarity
from jaccard import *
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
def textPreProcess(SourcePath1,SourcePath2):
    # 每两行的句子为一对构造矩阵
    # 1.统计每一行句子的单词数量，然后计算单词数量的分布，以便计算矩阵的长宽
    # 统计txt文件的行数
    file_raws = len(["" for line in open(SourcePath1, "r", encoding='utf8')])
    print(file_raws)
    text2 = open(SourcePath2, "r", encoding='utf8')
    couple = file_raws//2
    wordsAll = np.zeros((couple,3,3))  # 整个文档的矩阵
    label_1 = np.zeros(couple)  # 整个文档的句子对的标签（有多少句子对就有多少个标签）
    labels = label_1.astype(np.str) # 类型转换
    sen_length = []  # 统计每个句子的单词数量
    lines = []
    parse= []

    for line in text2.readlines():
        words3 = re.split(r' ', line)
        words4 = list(filter(lambda
                                 x: x != '\n' and x != '，' and x != '。' and x != '、' and x != '：' and x != '；' and x != '（' and x != '）',
                             words3))
        parse.append(words4)
    text2.close()
    csv_num = 1
    for raw_num in range(file_raws-1):
        if raw_num % 2 == 1:
            continue
        else:
            com_num = wordsCount  # 确定两个句子比较的次数
            # a = [[]for i in range(wordsAver)]
            a = np.zeros(shape=(3, 3))  # 初始化一个矩阵
            high = raw_num
            low = raw_num + 1
            for i in range(3):
                #print(parse[high * 3 + i])
                for j in range(3):

                   # print(parse[low * 3 + j])
                    a[j][i] = jaccard_similarity(parse[high * 3 + i], parse[low * 3 + j])
        wordsAll[csv_num-1] = a
        print(a)
        csv_num = csv_num + 1
    label = os.path.basename(SourcePath1)[:-8] #截取从头开始到倒数第八个字符之前
    for i in range(csv_num-1):
        labels[i] = label

    return wordsAll, labels
            # print(a)
            # dataframe = pd.DataFrame(a)
            # dataframe.to_csv(CountResult + '%d.csv' % (csv_num), index=False, sep=',', header=False)


def read_category():
    """读取分类目录，固定"""
    categories =['Action','Consequence', 'Subject']#'noConflict',
    cat_to_id = dict(zip(categories, range(len(categories)))) #{'Action':0, 'Consequence':1, 'Subject':3}
    return categories, cat_to_id

def process_file(filepath1,filepath2,cat_to_id):
    """将文件转换为id表示"""
    action_content, action_category = textPreProcess(filepath1 + 'Action_cut.txt',filepath2 + 'Action_parse.txt')
    Consequence_content, Consequence_category = textPreProcess(filepath1 + 'Consequence_cut.txt',filepath2 + 'Consequence_parse.txt')
    #noconflict_content, noconflict_category = textPreProcess(filepath1 + 'noConflict_cut.txt',filepath2 + 'noConflict_parse.txt')
    # Object_content, Object_category = textPreProcess(filepath1 + 'Object_cut.txt')
    Subject_content, Subject_category = textPreProcess(filepath1 + 'Subject_cut.txt',filepath2 + 'Subject_parse.txt')
    # contents = action_content
    contents = np.vstack((action_content, Consequence_content))#拼接矩阵
    #contents = np.vstack((contents, noconflict_content))
    # contents = np.vstack((contents, Object_content))
    contents = np.vstack((contents, Subject_content))
    labels = action_category
    labels = np.concatenate((action_category, Consequence_category), axis=0) #拼接数组
    #labels = np.concatenate((labels,noconflict_category), axis=0)
    # labels = np.concatenate((labels, Object_category), axis=0)
    labels = np.concatenate((labels, Subject_category), axis=0)
    #print(contents)
    label_id = []
    for i in range(len(labels)):
        label_id.append(cat_to_id[labels[i]]) #分配冲突类别
    #print(label_id)
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    # x_pad = kr.preprocessing.sequence.pad_sequences(contents, max_length)
    # y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示
    x_pad = contents
    y_pad = np.array(label_id) # 将标签转换为one-hot表示
    return x_pad, y_pad


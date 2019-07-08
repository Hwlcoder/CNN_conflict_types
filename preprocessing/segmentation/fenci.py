# coding=utf-8
#分词

import jieba
import sys
sys.path.append("../")
jieba.load_userdict("user_dict.txt")
import time
t1=time.time()

f1=open("dataset\\test_5.13_new\yuan\\test_5.13_3\Consequence.txt","r") #读取文本
f2=open("dataset\\test_5.13_new\my\\test_5.13_3\Consequence.txt","w")  #将结果保存到另一个文档中
for line in f1:
    wordlist=list(jieba.cut(line))
    tmp=''
    for word in wordlist:
        tmp += word + ' '
    tmp = tmp.replace('\n', '')
    f2.write(tmp+"\n")

f1.close()
f2.close()
t2=time.time()
print("分词完成，耗时："+str(t2-t1)+"秒。") #反馈结果


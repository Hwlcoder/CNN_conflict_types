import tensorflow as tf
import numpy as np
from textProcess3 import *
import math

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('model/modeluse_acc100/model.ckpt-300.meta')
  new_saver.restore(sess, 'model/modeluse_acc100/model.ckpt-300')

  my_data_path1 = "data/test/liu/test_3/"
  my_data_path2 = "data/test/my/test_parser3/"
  categories, cat_to_id = read_category()
  test_x, test_y = process_file(my_data_path1,my_data_path2,cat_to_id)
  print(test_y)
  evaluation_size = len(test_x)

  conflict_dict = {0: '行为冲突', 1: '后果冲突',  2: '主体冲突'}

  # 定义数据集的占位符
  # 验证图片输入张量
  eval_input_shape = (evaluation_size, 3, 3)
  # 创建输入验证数据的占位符
  eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
  # 创建一个批量验证结果的占位符
  eval_target = tf.placeholder(tf.int32, shape=evaluation_size)

  graph = tf.get_default_graph()
  x = graph.get_tensor_by_name("x:0")
  test_x = np.expand_dims(test_x, 3)
  test_dict = {x: test_x}
  logits = graph.get_tensor_by_name("logits_eval:0")

  test_preds = sess.run(logits, feed_dict=test_dict)


  output = []
  output = tf.argmax(test_preds, 1).eval()
  print(output)
  #for i in range(len(output)):
      #print("第", i + 1, "对的结果:" + conflict_dict[output[i]])

#coding=utf-8
import os
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES = 48 #48 classes
IMG_W = 30
IMG_H = 30
BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 10000 #
learning_rate = 0.0001 #

train_dir='D:/20180425/DeepFiData/train/'
logs_train_dir='D:/20180425/DeepFiData/logs/'

train, train_label = input_data.get_files(train_dir)
train_batch,train_label_batch=input_data.get_batch(train,
                                train_label,
                                IMG_W,
                                IMG_H,
                                BATCH_SIZE,
                                CAPACITY)
train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)
train_op = model.trainning(train_loss, learning_rate)
train__acc = model.evaluation(train_logits, train_label_batch)
summary_op = tf.summary.merge_all() #这个是log汇总记录

#产生一个会话
sess = tf.Session()
#产生一个writer来写log文件
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
 #产生一个saver来存储训练好的模型
saver = tf.train.Saver()
#所有节点初始化
sess.run(tf.global_variables_initializer())

#队列监控
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    #执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
                break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
        #每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
        if step % 50 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        #每隔2000步，保存一次训练好的模型
        if step % 2000 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()





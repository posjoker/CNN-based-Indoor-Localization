import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import model
import heapq

def get_one_image(train,row,col):
    suffix = str(row*6+col)+'/'+str(random.randint(2, 800))+'.jpg'
    img_dir = train+suffix
    image = np.array(Image.open(img_dir).resize([30, 30]))
    return image, img_dir
def evaluate_one_image(train_dir,row,col):
    image_array, img_dir=get_one_image(train_dir, row, col)
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASS = 48
        image = tf.cast(image_array,tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 30, 30, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASS)
        logit=tf.nn.softmax(logit)
        x=tf.placeholder(tf.float32, [30, 30, 3])
        #logs_train_dir = 'D:/20180418/logs/'
        logs_train_dir = 'D:/20180425/DeepFiData/logs/'
        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            '''
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1])
            '''
            #print(img_dir)
            #print(max_index)
            print("The ground truth is %d and the prediction is %d"%(row*6+col, max_index))

    argindex = np.array(prediction)[0].argsort()[-3:]
    argweight = prediction[0][argindex[:]]
    axis_x = list(map(lambda a: int(a/6), argindex))
    axis_y = list(map(lambda a: a % 6, argindex))
    squareweight = list(map(lambda a: np.square(a), argweight))
    centroid_x = (np.dot(np.array(axis_x), squareweight))/sum(squareweight)
    centroid_y = (np.dot(np.array(axis_y), squareweight))/sum(squareweight)
    err = np.sqrt(np.square(centroid_x-row)+np.square(centroid_y-col))
    print("The actual location is %d %d while predicted Location is %.6f %.6f"% (row, col, centroid_x, centroid_y))
    print("The prediction err is %.2f"% err)

    if max_index == (row*6+col):

        return 1, err
    else:
        return 0, err

if __name__ == '__main__':
    train_dir = 'D:/20180425/DeepFiData/train/'
    cnt = acc = avg = 0
    for i in range(0, 8):
        for j in range(0, 6):
            cnt += 1
            temp, err = evaluate_one_image(train_dir, i, j)
            acc += temp
            avg += err
    acc = acc/cnt
    avg = avg/cnt
    print("correct prediction rate is %.4f, and localization err is %.4f"%(acc, avg))



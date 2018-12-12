#2018-10-01 DongYoung Kim

import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.layers as layers

#데이터가 길어질 경우에 numpy를 통하여 csv파일 안에 데이터 값을 주어서 불러올 수 있다.
xy = np.loadtxt('./data-03-diabetes.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1] # n행 전체 / 뒤에서 1번째 까지
y_data = xy[:, [-1]] # n행 전체 / 마지막 -1번째 데이터만 가져온다

test_m = 100
x_train = x_data[:-test_m]
y_train = y_data[:-test_m]
x_test =  x_data[-test_m:]
y_test =  y_data[-test_m:]

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None,8]) # X데이터가 8개 이기때문에 8로 수정
Y = tf.placeholder(tf.float32, shape=[None,1])

hidden0 = layers.fully_connected(X,16) # 한 단계 더 깊은 layer
hidden1 = layers.fully_connected(hidden0,32) # 한 단계 더 깊은 layer  2^n 단위로 깊이가 깊어질 수록 neurons수를 2^n만큼 늘린다. ( 사람 뇌를 흉내 )
hypothesis = layers.fully_connected(hidden1,1, tf.nn.sigmoid) # W, b 알아서 다해줌

#hypothesis = layers.fully_connected(X,1, tf.nn.sigmoid) # W, b 알아서 다해줌
# W = tf.Variable(tf.random_normal([8,1]), name = 'weight')
# b = tf.Variable(tf.random_normal([1]), name = 'bias')
#
# # Hypothesis using sigmoid : tf.div(1. , 1. + tf.exp(tf.matmul(X,W) + b )
# hypothesis = tf.sigmoid(tf.matmul(X,W) + b)x

# cost/loss function (Logistic classfication cost)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + ( 1 - Y )*tf.log( 1 - hypothesis ))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False 보통의 기준은 0.5로 준다.
predicted = tf.cast(hypothesis > 0.5 , dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y) , dtype = tf.float32))
# true = 1 false = 0

# Launch graph
with tf.Session() as sess:
    # # Initialize TensorFlow variables
    # sess.run(tf.global_variables_initializer())
    # print(is_saved)
    saver = tf.train.Saver()
    # is_saved = True
    is_saved = os.path.isfile("./checkpoint") # 파일 존재 유무 확인
    if is_saved:
        saver.restore(sess, './my_model.ckpt')  # 체크 포인트
    else:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
    feed = {X:x_data,Y:y_data}
    for step in range(10001):
        cost_val , _ = sess.run([cost,train], feed_dict=feed)
        if step % 200 == 0:
            print(step,cost_val)
            saver.save(sess, './my_model.ckpt') # 체크 포인트

    # Accuracy report
    h, c , a = sess.run([hypothesis,predicted,accuracy],feed_dict={X:x_data,Y:y_data})

    print("\nHypothesis : \n" , h ,"\nCorrect (Y) : \n" , c , "\nAccuracy : " , a )

# acc_test = sess.run(accuracy , {X:x_test , Y:y_test})
# print('acc_test',acc_test)
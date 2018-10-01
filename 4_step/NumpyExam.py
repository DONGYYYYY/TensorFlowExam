#2018-10-01 DongYoung Kim

import tensorflow as tf
import numpy as np

#데이터가 길어질 경우에 numpy를 통하여 csv파일 안에 데이터 값을 주어서 불러올 수 있다.
xy = np.loadtxt('./data-03-diabetes.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1] # n행 전체 / 뒤에서 1번째 까지
y_data = xy[:, [-1]] # n행 전체 / 마지막 -1번째 데이터만 가져온다


# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None,8]) # X데이터가 8개 이기때문에 8로 수정
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([8,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Hypothesis using sigmoid : tf.div(1. , 1. + tf.exp(tf.matmul(X,W) + b )
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

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
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    feed = {X:x_data,Y:y_data}
    for step in range(10001):
        cost_val , _ = sess.run([cost,train], feed_dict=feed)
        if step % 200 == 0:
            print(step,cost_val)

    # Accuracy report
    h, c , a = sess.run([hypothesis,predicted,accuracy],feed_dict={X:x_data,Y:y_data})

    print("\nHypothesis : \n" , h ,"\nCorrect (Y) : \n" , c , "\nAccuracy : " , a )
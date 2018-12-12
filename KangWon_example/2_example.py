import tensorflow as tf
import numpy as np

tf.random.set_random_seed(0) # random값을 일정하게 주기 위한 seed

alpha = 3
beta = 2

# x_train = np.array([1,2,3])
# y_train = x_train * alpha + beta

x_train = tf.placeholder(tf.float32, shape = [None])
y_train = tf.placeholder(tf.float32 , shape = [None])

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train*W+b
cost = tf.reduce_mean(tf.square(hypothesis-y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 초기화 최초 1회 필요

for step in range(2001):
    # sess.run(train)
    sess.run(train,feed_dict={x_train:[1,2,3],y_train:[2,4,6]})
    #cost_val,_ = sess.run([cost,train],feed_dict={x_train:[1,2,3],y_train:[2,4,6]})
    if step % 200 == 0:
        print(step,sess.run(cost,feed_dict={x_train:[1,2,3],y_train:[2,4,6]}),sess.run(W),sess.run(b)) # placeholder가 들어간 식을 사용할 경우에는 feed_dict를 통해서 data 값을 지정해 주어야 함.
        #W_val , B_val = sess.run([W,b])
        # print(step, cost_val, W_val, B_val)

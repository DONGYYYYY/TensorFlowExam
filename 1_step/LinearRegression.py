# 2018-9-28 DongYoung Kim

import tensorflow as tf
# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

#x =1 일때 y = 1 x = 2일때 y = 2 와 같은 데이터.

w = tf.Variable(tf.random_normal([1]),name = 'weight')
# tensorflow가 자체적으로 변경시키는 변수 학습하는 과정에서 스스로 변경 [1] = 값을 1개를 사용한다는 것을 의미
# trainable variable.
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train*w + b

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#reduce_mean = 평균을 내주는 작업을 하는 메소드

#Minimize 중요 !
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)
#train -> 그래프의 이름 (node)

#Launch the graph in a session
sess = tf.Session()
#Initializes global variables in the graph.
sess.run(tf.global_variables_initializer()) # 반드시 사용하기 전에 초기화 작업을 해야한다.

#Fit the Line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(cost),sess.run(w) , sess.run(b))


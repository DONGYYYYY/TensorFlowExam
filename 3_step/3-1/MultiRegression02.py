# 2018-10-01 DongYoung Kim
#Use Matrix

import tensorflow as tf

x_data = [[73.,80.,75.],[93.,88.,93.],[89.,91.,90.],[96.,98.,100.],[73.,66.,70.]]
y_data = [[152.],[185.],[180.],[196.],[142.]]

X = tf.placeholder(tf.float32 , shape=[None,3]) # instance = None(n개의 값을 뜻함) / Variable = 3
Y = tf.placeholder(tf.float32 , shape=[None,1]) # instance = None(n개의 값을 뜻함) / Y = 1

W = tf.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Hypothesis (Matrix)
hypothesis = tf.matmul(X,W) + b
# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
# range값의 크기가 커질 수록 오차의 편차는 적어진다.,
for step in range((int)(10e+4+1)): # 10^5
    cost_val, hy_val, _ = sess.run(
        [cost,hypothesis,train],feed_dict={X:x_data,Y:y_data})
    if step % 10e+3 ==0: # 10^4
        print(step,"Cost : ",cost_val,"\nPrediction:\n",hy_val)
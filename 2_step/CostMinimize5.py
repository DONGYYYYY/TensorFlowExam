# 2018-9-28 DongYoung Kim

import tensorflow as tf
X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(5.)
hypothesis = X * W

gradient = tf.reduce_mean((W*X-Y)*X)*2

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#일치

gvs = optimizer.compute_gradients(cost) # 코스트에 맞는 gradient를 계산을 하여 gvs에 return
apply_gradients = optimizer.apply_gradients(gvs) # 값을 apply

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient,W,gvs])) # gvs 출력 시 gradient와 weight값이 순차적으로 출력
    sess.run(apply_gradients)

#tensorflow가 돌린 gradient값과 직접 식으로 계산한 gradient값이 일치하는 것을 알 수 있음.
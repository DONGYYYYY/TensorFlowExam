import tensorflow as tf

X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(-3.0)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
# learning_rate = 0.1
# gradient = tf.reduce_mean((W * X - Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent)
#와 같은 역할을 함.

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    if step % 10 == 0:
        print(int(step/10) ," : ", sess.run(W))
        sess.run(train)

        
# import tensorflow as tf
# x_data = [ 1 , 2 , 3 ]
# y_data = [ 1 , 2 , 3 ]
#
# W = tf.Variable(tf.random_normal([1]), name='weight')
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# W = tf.Variable(-3.0)
#
# hypothesis = X * W
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(100):
#     if step % 10 == 0:
#         print(int(step/10) ," : ", sess.run(W))
#         sess.run(train , feed_dict={ X:x_data , Y: y_data})
# 2018-10-01 DongYoung Kim

import tensorflow as tf

filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'],#파일 수는 증가 할 수 있음.
                                                  shuffle=False , name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the decoded result
record_defaults = [[0.],[0.],[0.],[0.]] # 데이터 type을 정해주기 위한 record_defaults변수 정의
xy = tf.decode_csv(value,record_defaults=record_defaults)

# collect batches of csv in
train_x_batch , train_y_batch = tf.train.batch([xy[0:-1],xy[-1:]],batch_size=10) # batch_size = 한번에 가져오는 양

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32 , shape=[None,3])
Y = tf.placeholder(tf.float32 , shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

# Hypothesis
hypothesis = tf.matmul(X,W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Start populating the filename queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess , coord=coord)

for step in range(2001):
    x_batch , y_batch = sess.run([train_x_batch,train_y_batch])
    cost_val , hy_val , _ = sess.run([cost,hypothesis,train],feed_dict=
    {X : x_batch , Y : y_batch})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val , "\nPrediction : \n", hy_val)

coord.request_stop()
coord.join(threads)
# 2018-10-01 DongYoung Kim

import tensorflow as tf

filename_queue = tf.train.string_input_producer(['./data-03-diabetes.csv'],#파일 수는 증가 할 수 있음.
                                                  shuffle=False , name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the decoded result
record_defaults = [[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]] # 데이터 type을 정해주기 위한 record_defaults변수 정의
xy = tf.decode_csv(value,record_defaults=record_defaults)

# collect batches of csv in
train_x_batch , train_y_batch = tf.train.batch([xy[0:-1],xy[-1:]],batch_size=18) # batch_size = 한번에 가져오는 양

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32 , shape=[None,8])
Y = tf.placeholder(tf.float32 , shape=[None,1])

W = tf.Variable(tf.random_normal([8,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

# Hypothesis using sigmoid : tf.div(1. , 1. + tf.exp(tf.matmul(X,W) + b )
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

# cost/loss function (Logistic classfication cost)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + ( 1 - Y )*tf.log( 1 - hypothesis ))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False 보통의 기준은 0.5로 준다.
predicted = tf.cast(hypothesis > 0.5 , dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y) , dtype = tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(10001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val , _ = sess.run([cost,train], feed_dict={X:x_batch,Y:y_batch})
        if step % 200 == 0:
            print(step,cost_val)

    # Accuracy report
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    h, c , a = sess.run([hypothesis,predicted,accuracy],feed_dict={X:x_batch,Y:y_batch})

    print("\nHypothesis : \n" , h ,"\nCorrect (Y) : \n" , c , "\nAccuracy : " , a )
    coord.request_stop()
    coord.join(threads)
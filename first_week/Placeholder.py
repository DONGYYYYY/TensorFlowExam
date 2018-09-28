import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')
X = tf.placeholder(tf.float32 , shape = [None]) # 개수를 제한할 수 있음 shape를 사용하여.
Y = tf.placeholder(tf.float32 , shape = [None])

#Our hypothesis XW + b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

#학습 시작
# Fit the line with new training data
for step in range(4000):
    cost_val , W_val, b_val, _ =  sess.run([cost,W,b,train],
    feed_dict={X: [1,2,3,4,5] , Y: [2.1,3.1,4.1,5.1,6.1]}) # 학습 데이터 ( X , Y )를 준다.

    if step % 100 == 0:
        print(step , cost_val , W_val , b_val)
        #step / cost / w / b 값 순서대로 화면에 출력
#학습 종료

#Testing our model
print(sess.run(hypothesis, feed_dict={X :[5]}))
print(sess.run(hypothesis, feed_dict={X :[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5,3.5]}))
#학습이 없을 경우에는 값이 일정하지 않음.
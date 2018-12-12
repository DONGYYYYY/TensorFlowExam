import tensorflow as tf
import tensorflow.contrib.layers as layers

x_data = [[73,80,75],[93,88,93],[89,91,90],[99,98,100],[73,66,70.0]]
y_data = [[152],[185],[180],[196],[142]]

X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,1])

hypothesis = layers.fully_connected(X,1) # W, b 알아서 다해줌
# W = tf.Variable(tf.random_normal([3,1]))
# b = tf.Variable(tf.random_normal([1]))
# hypothesis = tf.matmul(X,W) + b
# hypothesis = tf.nn.reLu(hypothesis)

cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

#기능( = 모델 = 그래프 ) 구현 긑, 이제 사용하자
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val,hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={X:x_data,Y:y_data})
    if step % 20==0:
        print(cost_val,hy_val)

import tensorflow as tf

a = tf.placeholder(tf.float32) # 함수의 인자
b = tf.placeholder(tf.float32)

c = a + b
d = a * b

sess = tf.Session()

c_,d_ = sess.run([c,d],feed_dict = {a : 3 , b : 4.5})
# 계산 값 , feed_dict = data
print('c_:',c_,"\td_:",d_)


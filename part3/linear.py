import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial)

# dataset
xx = np.random.randint(0,1000,[1000,3])/1000.
yy = xx[:,0] * 2 + xx[:,1] * 1.4 + xx[:,2] * 3

# model
x = tf.placeholder(tf.float32, shape=[None, 3])
y_ = tf.placeholder(tf.float32, shape=[None])
W1 = weight_variable([3, 1])
y = tf.matmul(x, W1)

# training and cost function
cost_function = tf.reduce_mean(tf.square(tf.squeeze(y) - y_))
train_function = tf.train.AdamOptimizer(1e-2).minimize(cost_function)

# create a session
sess = tf.Session()

# train
sess.run(tf.initialize_all_variables())
for i in range(10000):
    sess.run(train_function, feed_dict={x:xx, y_:yy})
    if i % 1000 == 0:
        print(sess.run(cost_function, feed_dict={x:xx, y_:yy}))
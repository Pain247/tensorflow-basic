from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


X_train = np.asarray([1.47012, 1.50210, 1.53320, 1.5810, 1.6310, 1.6501, 1.6801, 1.7020, 1.7301, 1.7502, 1.7802, 1.8003, 1.8303])
Y_train = np.asarray([ 49.03, 50.03, 51.01,  54.03, 58.02, 59.03, 60.01, 62.03, 63.03, 64.03, 66.03, 67.30, 68.30])
d = X_train.shape[0]

random = np.random
learning_rate = 0.008
display_step = 100
X = tf.placeholder("float")
Y = tf.placeholder("float")
w= tf.Variable(random.randn())
b = tf.Variable(random.randn())
model = tf.add(tf.multiply(X, w), b)   # Linear model
loss = tf.reduce_sum(tf.pow(model-Y, 2))/(2*d) #squared error
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)   # Gradient descent
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(100000):
        for (x, y) in zip(X_train, Y_train):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if (i+1) % display_step == 0:
            c = sess.run(loss, feed_dict={X: X_train, Y:Y_train})
            print("Epoch:", '%04d' % (i+1), "loss=", "{:.8f}".format(c), \
                "w=", sess.run(w), "b=", sess.run(b))
   
    training_cost = sess.run(loss, feed_dict={X: X_train, Y: Y_train})
    print("Training loss=", training_loss, "w=", sess.run(w), "b=", sess.run(b), '\n')
    y_test = 1.55*sess.run(w) + sess.run(b)
    print("Predict weight of person with height 1.55m: %.2f (kg)" %(y_test))
    plt.plot(X_train, Y_train, 'ro', label='Original data')
    plt.plot(X_train, sess.run(w) * X_train + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

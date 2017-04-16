from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 100
display_step = 1

n_hidden_1 = 256 # 1st layer 
n_hidden_2 = 256 # 2nd layer 
n_input = 784 # MNIST data input (shape: 28*28)
n_classes = 10 #  10 classes


x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# w & b object
w={
    'h1' : tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
} 
b={
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([n_classes]))
}   

# define model
def NNmodel(x,w,b):
    #layer1
    l1 = tf.add(tf.matmul(x, w['h1']),b['b1'])
    l1 = tf.nn.relu(l1)
    #layer2
    l2 = tf.add(tf.matmul(l1,w['h2']),b['b2'])
    l2 = tf.nn.relu(l2)
    #output
    out = tf.matmul(l2,w['out']) + b['out']
    return out

#construct model

model = NNmodel(x,w,b)

#Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#Init the variable
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #training
    for i in range(20):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for j in range(total_batch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,loss],feed_dict={x : x_batch, y: y_batch})
            avg_loss += c/total_batch
        if(i % display_step==0):
            print ("Epoch: ",'%0.4d'%(i+1), "loss =","{:.8f}".format(avg_loss))
    pred = tf.equal(tf.argmax(model,1), tf.arg_max(y,1))
    accuracy = tf.reduce_mean(tf.cast(pred,"float"))
    print ("Accuracy: ", accuracy.eval({x: mnist.test.images, y : mnist.test.labels}))        
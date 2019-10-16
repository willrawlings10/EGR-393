#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Assignment: Multi-Layer Perceptron (MLP)

# ### Multi-layer Perceptron
# 
# Build a 2-layer MLP for MNIST digit classfication. Feel free to play around with the model architecture and see how the training time/performance changes, but to begin, try the following:
# 
# Image (784 dimensions) -> fully connected layer (500 hidden units)  -> nonlinearity (ReLU) -> fully connected layer (100 hidden units) -> nonlinearity (ReLU) -> fully connected (10 hidden units) -> softmax

# Make sure to print out your accuracy on the test set at the end.


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange         
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
Z= tf.Variable(tf.truncated_normal([500, 100], stddev=0.1))
Y = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b = tf.Variable(tf.truncated_normal([10], stddev=0.1))
scores = (X @ W @ Z @ Y) + b
p_scores = tf.nn.softmax(scores)

y = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in trange(100):
    for which_batch in range(550):
        batch_xs = mnist.train.images[which_batch*100:(which_batch+1)*100]
        batch_ys = mnist.train.labels[which_batch*100:(which_batch+1)*100]
        sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})
        


correct_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Test accuracy: {0}'.format(sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})))

weights = sess.run(W)

fig, ax = plt.subplots(1, 10, figsize=(20, 2))

for digit in range(10):
    ax[digit].imshow(weights[:,digit].reshape(28,28), cmap='gray')


# In[ ]:
"""
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis =1 )
x_test = tf.keras.utils.normalize(x_test, axis =1 )
model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(100,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))

model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 4)

"""


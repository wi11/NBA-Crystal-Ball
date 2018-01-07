import tensorflow as tf
import csvparser

#paramaters for machine learning model
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

x = tf.placeholder(tf.float32, [None, 6]) #shape of each feature vector is 6 because there are 6 features
y = tf.placeholder(tf.float32, [None, 2]) #there are 2 classes: Win or lose: 0 or 1

#linear regression model y = Wx + b
W = tf.Variable(tf.zeros[6, 2])         #6 weights for each of the 6 features
b = tf.Variable(tf.zeros[2])            #2 biases for each class

#machine learning model for prediction function
pred = tf.nn.softmax(tf.matmul(x, W) + b)

#minimizes error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indicies=1))

# train using gradient descent to minimize cost
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

#collect data
parsed = csvparser.parse()
features = parsed[0]
outcomes = parsed[1]


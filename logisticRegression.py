import tensorflow as tf
import csvparser

#paramaters for machine learning model
learning_rate = 0.01
num_epochs = 30
batch_size = 100
display_step = 2

#shape of each feature vector is 6 because there are 6 features
x = tf.placeholder(tf.float32, [None, 6]) 
#there are 2 classes: Win or lose: 0 or 1
y = tf.placeholder(tf.float32, [None, 2]) 

#linear regression model y = Wx + b
W = tf.Variable(tf.zeros([6, 2]))         #6 weights for each of the 6 features
b = tf.Variable(tf.zeros([1]))            #2 biases for each class

#machine learning model for prediction function
pred = tf.nn.softmax(tf.matmul(x, W) + b)

#minimizes error using cross entropy
cost = -tf.reduce_sum(y*tf.log(pred))

# train using gradient descent to minimize cost
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

#collect training and testing data
teams = {}
training_data = csvparser.parse_data(teams, "2016_Box_Scores.csv")
training_features = training_data[0]
training_outcomes = training_data[1]
testing_data = csvparser.parse_data({}, "2016_Playoffs_Box_Scores.csv")
testing_features = testing_data[0]
testing_outcomes = testing_data[1]

#initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(num_epochs):
    #optimize at each step
    sess.run(optimizer, feed_dict={x:training_features, y:training_outcomes})

#test the accuracy of this model by seeing how well it predicts 2016 playoff games
#compare if the predicted label and real label are the same
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), "float"))

accuracy_value = sess.run(accuracy, feed_dict={x:testing_features, y:testing_outcomes})
print accuracy_value

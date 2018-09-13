import numpy as np
import tensorflow as tf
import pandas as pd
import time

# Disable chain indexing warning - Not highly advised unless you know what your doing.
pd.options.mode.chained_assignment = None

train_dataset = pd.read_csv('train.csv', sep=',')
train_values = list(train_dataset.columns.values)

test_dataset = pd.read_csv('test.csv', sep=',')
test_values = list(test_dataset.columns.values)

X_train = train_dataset[[train_values[2], train_values[4], train_values[5], train_values[7], train_values[10]]]
Y_train = train_dataset[train_values[1]]

X_test = test_dataset[[test_values[1], test_values[3], test_values[4], test_values[6], test_values[9]]]
Y_test = test_dataset[test_values[1]]

# Tweak columns
X_train['Sex'].replace(["female", "male"], [0,1], inplace=True)
X_train['Age'].replace(np.nan, 0, inplace=True)
X_train['Cabin'].replace(np.nan, "0", inplace=True)
X_train['Cabin'] = X_train['Cabin'].astype(str)
X_train['Cabin'] = X_train['Cabin'].apply(lambda x: sum(ord(i) for i in x))

X_test['Sex'].replace(["female", "male"], [0,1], inplace=True)
X_test['Age'].replace(np.nan, 0, inplace=True)
X_test['Cabin'].replace(np.nan, "0", inplace=True)
X_test['Cabin'] = X_test['Cabin'].astype(str)
X_test['Cabin'] = X_test['Cabin'].apply(lambda x: sum(ord(i) for i in x))

X_train = np.array(X_train)
Y_train = np.reshape(np.array(Y_train), (-1,1))

X_test = np.array(X_test)
Y_test = np.reshape(np.array(Y_test), (-1,1))

# Shuffle Data
indices = np.random.permutation(len(X_train))
X_input = X_train[indices]
y_output = Y_train[indices]

input_layer = len(X_input[0])
hidden_layer = 14
output_layer = 1

# Tensor setup
X_data = tf.placeholder(tf.float32, shape=[None, input_layer], name="x-inputdata")
y_target = tf.placeholder(tf.float32, shape=[None, output_layer], name="y-targetdata")

weight_one = tf.Variable(tf.random_uniform([input_layer, hidden_layer], -1, 1), name="Weight_One")
weight_two = tf.Variable(tf.random_uniform([hidden_layer, output_layer], -1, 1), name="Weight_Two")

bias_one = tf.Variable(tf.zeros([hidden_layer]), name="Bias_One")
bias_two = tf.Variable(tf.zeros([output_layer]), name="Bias_Two")

with tf.name_scope("layer2") as scope:
	synapse0 = tf.sigmoid(tf.matmul(X_data, weight_one) + bias_one, name="Synapse0")

with tf.name_scope("layer3") as scope:
	hypothesis = tf.sigmoid(tf.matmul(synapse0, weight_two) + bias_two, name="Synapse1")

with tf.name_scope("cost") as scope:
	cost = tf.reduce_mean(( (y_target * tf.log(hypothesis)) + ((1 - y_target) * tf.log(1.0 - hypothesis)) ) * -1, name="Cost")

with tf.name_scope("train") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	
	writer = tf.summary.FileWriter("./logs/titanic", sess.graph)

	sess.run(init)

	t_start = time.clock()
	for i in range(10000000):
		sess.run(train_step, feed_dict={X_data : X_input, y_target : y_output })
		if i % 1000 == 0:
			print("Epoch ", i)
			print("Hypothesis ", sess.run(hypothesis, feed_dict={X_data : X_input, y_target : y_output}))
			print("Weight 1 ", sess.run(weight_one))
			print("Bias 1 ", sess.run(bias_one))
			print("Weight 2 ", sess.run(weight_two))
			print("Bias 2 ", sess.run(bias_two))
			print("cost ", sess.run(cost, feed_dict={X_data : X_input, y_target : y_output}))
	t_end = time.clock()
	print("Elapsed time ", (t_end - t_start))
	
	# Save to output due to training being complete
	save_path = saver.save(sess, "./saves/titanic.ckpt")
	total = 0
	correct = 0
	print("Testing Titanic survivors.")
	for i in range(len(X_input)):
		actual = Y_train[i]
		predicted = np.rint(sess.run(hypothesis, feed_dict={X_data : [X_train[i]]}))
		if actual == predicted:
			correct += 1
		total += 1
		print("Actual: ", actual, "Predicted: ", predicted)
	print("Correct {} out of {}, Percentage {}".format(correct, total, (correct/total)*100))

	# Actual predicting
	prediction = pd.DataFrame(columns=["PassengerID", "Survived"])
	for i in range(len(X_test)):
		prediction = prediction.append(pd.DataFrame([[test_dataset[test_values[0]][i], int(np.rint(sess.run(hypothesis, feed_dict={X_data : [X_test[i]]}))[0][0])]], columns=["PassengerID", "Survived"]), ignore_index=True)
	prediction.to_csv("prediction.csv", sep=',', index=False)		

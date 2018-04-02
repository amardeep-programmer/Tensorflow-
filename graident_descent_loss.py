import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

sess = tf.Session()

x_value = np.random.normal(1,0.1,100)

y_value =np.repeat (10.,100)

x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1],dtype=tf.float32)



X = tf.Variable(tf.random_normal(shape=[1]))


output = tf.multiply(x_data, X)

loss_function = tf.square (output - y_target)


init = tf.global_variables_initializer()

sess.run(init)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)

training_step = optimizer.minimize(loss_function)

for i in range(100):
	index = np.random.choice(100)
	rand_x =[x_value[index]]
	rand_y = [y_value[index]]

	sess.run(training_step, feed_dict = {x_data :rand_x, y_target:rand_y})
	if (i+1)%10==0:

		print('Step #' + str(i+1) + ' X= ' + str(sess.run(X)))
		print('Loss = ' + str(sess.run(loss_function, feed_dict={x_data:rand_x, y_target: rand_y})))

	

writer = tf.summary.FileWriter('/home/amardeep/tensorflow_learning',sess.graph)

writer.close()
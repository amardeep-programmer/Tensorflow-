import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

sess = tf.Session()

x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. +tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)

x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500,], 1.)

hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. -target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)

weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)


x_value = sess.run(x_vals)

x_array = sess.run(x_vals)
plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss')
plt.plot(x_array, phuber1_y_out, 'k-.', label='P-Huber Loss (0.25)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.title("loss functions plot")
plt.grid()
plt.show()
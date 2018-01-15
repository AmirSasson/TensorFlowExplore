""" dsfd """
import tensorflow as tf

HELLO = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(HELLO))

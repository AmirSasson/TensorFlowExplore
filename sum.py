""" sum """
import tensorflow as tf

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)
node3 = tf.add(node1, node2)

sess = tf.Session()
print(sess.run(node3, {node1: [1, 2, 3, 4, 5], node2: [6, 7, 8, 9, 10]}))

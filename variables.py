""" sum """
import tensorflow as tf

W = tf.Variable([100], dtype=tf.float32)
b = tf.Variable([-80], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = tf.add(tf.multiply(W, x), b)  # ~ W*x + b
y = tf.placeholder(tf.float32)

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})


curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

# print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
# print(sess.run(node3, {node1: [1, 2, 3, 4, 5], node2: [6, 7, 8, 9, 10]}))

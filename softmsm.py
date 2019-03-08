import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



a = tf.Variable([2])
YY = tf.one_hot(a, 10)

logits = tf.Variable(tf.random_uniform(shape = [1, 10]) * tf.constant(0.001))

batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = YY, logits = logits)


aa = tf.reduce_sum(-1 * YY * tf.log(tf.nn.softmax(logits)), reduction_indices=True)


init = tf.global_variables_initializer()

with tf.Session() as s:
    s.run(init)
    # print(s.run(batch_xentropy))
    print(s.run(logits))
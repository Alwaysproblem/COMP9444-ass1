import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


x = [[1, 2, 3, 4, 5, 4, 3, 2, 1, 0]]

y = [[0, 1, 0, 0]]

X = tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_input")


Y = tf.placeholder(dtype=tf.float32, shape=[None, 4],
                          name="image_target_onehot")

layersize = 4

b = tf.Variable(tf.random_uniform(shape=[layersize, 1]), name = "bias-1", dtype=tf.float32)
w = tf.Variable(tf.random_uniform(shape=[layersize, int(X.shape[1])]), name="W-1")
logits = tf.matmul(w, X, transpose_b= True) + b
preds = tf.nn.softmax(logits, axis = 0)
Y_label = tf.transpose(Y)
batch_xentropy = tf.reduce_sum(-1 * Y_label * tf.log(tf.nn.softmax(logits)), reduction_indices=True)
batch_loss = tf.reduce_mean(batch_xentropy)


# b_t = tf.Variable(tf.random_uniform(shape=[1, layersize]), name = "bias-1", dtype=tf.float32)
# w_t = tf.Variable(tf.random_uniform(shape=[int(X.shape[1]), layersize]), name="W-1")

b_t = tf.transpose(b)
w_t = tf.transpose(w)

logits_t = tf.matmul(X, w_t) + b_t
preds_t = tf.nn.softmax(logits_t)
batch_xentropy_t = tf.reduce_sum(-1 * Y * tf.log(tf.nn.softmax(logits_t)), reduction_indices=True)
batch_loss_t = tf.reduce_mean(batch_xentropy_t)


init = tf.global_variables_initializer()
with tf.Session() as s:
    s.run(init)
    # print(s.run([w, w_t]))
    # print(s.run([b, b_t]))
    # print(s.run(logits, feed_dict={X:x, Y:y}))
    # print(s.run(logits_t, feed_dict={X:x, Y:y}))
    print(s.run(preds, feed_dict={X:x, Y:y}))
    print(s.run(preds_t, feed_dict={X:x, Y:y}))
    # print(s.run(Y_label, feed_dict={X:x, Y:y}))
    # print(s.run(batch_loss_t, feed_dict={X:x, Y:y}))
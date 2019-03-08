"""
All tensorflow objects, if not otherwise specified, should be explicity
created with tf.float32 datatypes. Not specifying this datatype for variables and
placeholders will cause your code to fail some tests.

You do not need to import any other libraries for this assignment.

Along with the provided functional prototypes, there is another file,
"train.py" which calls the functions listed in this file. It trains the
specified network on the MNIST dataset, and then optimizes the loss using a
standard gradient decent optimizer. You can run this code to check the models
you create in part II.
"""

import tensorflow as tf

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" PART I """


def add_consts():
    """
    EXAMPLE:
    Construct a TensorFlow graph that declares 3 constants, 5.1, 1.0 and 5.9
    and adds these together, returning the resulting tensor.
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.constant(5.9)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)

    return af


def add_consts_with_placeholder():
    """ 
    Construct a TensorFlow graph that constructs 2 constants, 5.1, 1.0 and one
    TensorFlow placeholder of type tf.float32 that accepts a scalar input,
    and adds these three values together, returning as a tuple, and in the
    following order:
    (the resulting tensor, the constructed placeholder).
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.placeholder(tf.float32, name = 'c3')

    af = c1 + c2 + c3

    return af, c3


def my_relu(in_value):
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """

    zero = tf.constant(0, dtype=tf.float32)

    out_value = tf.maximum(zero, in_value)

    return out_value


def my_perceptron(x):
    """
    Implement a single perception that takes four inputs and produces one output,
    using the RelU activation function you defined previously.

    Specifically, implement a function that takes a list of 4 floats x, and
    creates a tf.placeholder the same length as x. Then create a trainable TF
    variable that for the weights w. Ensure this variable is
    set to be initialized as all ones.

    Multiply and sum the weights and inputs following the peceptron outlined in the
    lecture slides. Finally, call your relu activation function.
    hint: look at tf.get_variable() and the initalizer argument.
    return the placeholder and output in that order as a tuple

    Note: The code will be tested using the following init scheme
        # graph def (your code called)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # tests here

    """

    # X = tf.get_variable('x', shape=(len(x), 1), dtype=tf.float64)
    # print(x)
    i = tf.placeholder(tf.float32, shape=[4,], name = "X")
    X = tf.expand_dims(i, 1)

    # i = tf.placeholder(tf.float32, shape=[1, x], name = "X")
    W = tf.get_variable(
        'W',
        shape=(1, x), 
        dtype=tf.float32, 
        initializer=tf.constant_initializer(1)
    )
    # X = tf.transpose(i)
    # X = i
    Y = tf.matmul(W, X)

    out = my_relu(Y)

    return i, out


""" PART II """
fc_count = 0  # count of fully connected layers. Do not remove.


def input_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")


def target_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")




def onelayer(X, Y, layersize=10):
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    # b = tf.Variable(1.0, name = "bias")
    # b = tf.Variable(tf.random_uniform(shape=[layersize, 1]), name = "bias-1", dtype=tf.float32)
    b = tf.Variable(tf.random_uniform(shape=[1, layersize]), name = "bias-1", dtype=tf.float32)
    
    # w = tf.get_variable("W_one", dtype = tf.float32, , initializer=tf.random_uniform_initializer)
    w = tf.Variable(tf.random_uniform(shape=[int(X.shape[1]), layersize]), name="W-1")
    # w = tf.Variable(tf.random_uniform(shape=[layersize, int(X.shape[1])]), name="W-1")

    # print(X.shape)

    # Z = tf.matmul(w, X, transpose_a=True, transpose_b=True) + b

    # logits = tf.transpose(Z)
    # logits = tf.matmul(w, X, transpose_b= True) + b
    logits = tf.matmul(X, w) + b

    # print(logits.shape)

    preds = tf.nn.softmax(logits)

    # y = tf.transpose(Y)

    # print(y.shape)

    batch_xentropy = tf.reduce_sum(-1 * Y * tf.log(tf.nn.softmax(logits)), reduction_indices=True)

    # print(batch_xentropy.shape)

    batch_loss = tf.reduce_mean(batch_xentropy)

    return w, b, logits, preds, batch_xentropy, batch_loss


def twolayer(X, Y, hiddensize=30, outputsize=10):
    """
    Create a Tensorflow model for a Neural Network with one hidden layer

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """

    w1 = tf.Variable(tf.random_uniform(shape = [int(X.get_shape()[1]), hiddensize]) * tf.constant(0.001), dtype=tf.float32)
    w2 = tf.Variable(tf.random_uniform(shape = [hiddensize, outputsize]) * tf.constant(0.001), dtype=tf.float32)

    # b1 = tf.constant([1] * hiddensize, name = "bias1", dtype=tf.float32)
    # b2 = tf.constant([1] * outputsize, name = "bias2", dtype=tf.float32)

    b1 = tf.Variable(tf.random_uniform(shape=[1, hiddensize]), name = "bias1", dtype=tf.float32)
    b2 = tf.Variable(tf.random_uniform(shape=[1, outputsize]), name = "bias2", dtype=tf.float32)

    Z1 = tf.matmul(X, w1) + b1
    A1 = my_relu(Z1)
    logits = tf.matmul(A1, w2) + b2
    preds = tf.nn.softmax(logits)

    batch_xentropy = tf.reduce_sum(-1 * Y * tf.log(preds), reduction_indices=True)
    batch_loss = tf.reduce_mean(batch_xentropy)

    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss


def convnet(X, Y, convlayer_sizes=[10, 10], \
            filter_shape=[3, 3], outputsize=10, padding="same"):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    hints:
    1) consider tf.layer.conv2d
    2) the final layer is very similar to the onelayer network. Only the input
    will be from the conv2 layer. If you reshape the conv2 output using tf.reshape,
    you should be able to call onelayer() to get the final layer of your network
    """

    conv1 = tf.layers.conv2d(X, convlayer_sizes[0], filter_shape, padding= padding, activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(conv1, convlayer_sizes[1], filter_shape, padding= padding, activation=tf.nn.relu)

    b = tf.Variable(tf.random_uniform(shape=[1, outputsize]), name = "bias", dtype=tf.float32)
    # b = tf.constant([1] * outputsize, name = "bias-cov", dtype=tf.float32)
    # b = tf.ones([1, outputsize], name = "bias-cov", dtype=tf.float32)
    # w = tf.get_variable("W_con", dtype = tf.float32, shape=[conv2.shape[-1] * conv2.shape[-2] * conv2.shape[-3], outputsize], initializer=tf.random_uniform_initializer)
    w = tf.Variable(tf.random_uniform(shape=[int(conv2.shape[-1]) * int(conv2.shape[-2]) * int(conv2.shape[-3]), outputsize]), name = "W-con", dtype=tf.float32)

    x = tf.reshape(conv2, shape = [-1, int(conv2.shape[-1] * conv2.shape[-2] * conv2.shape[-3])])
    # x = tf.transpose(x)

    logits = tf.matmul(x, w) + b
    preds = tf.nn.softmax(logits)

    batch_xentropy = tf.reduce_sum(-1 * Y * tf.log(tf.nn.softmax(logits)), reduction_indices=True)
    batch_loss = tf.reduce_mean(batch_xentropy)

    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss


def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.

    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    
    # with tf.Session() as a:
    #     print("the X is ", a.run(X, feed_dict={X: batch[0]}))
    return train_result, loss, summary


# def main():
#     # graph def (your code called)
#     # in_value = tf.placeholder(tf.float32)
#     # max_v = my_relu(in_value)
#     # out_value = my_relu(in_value)
#     x = [3.0, 2.0, 1.0, 6.0]

#     X, out = my_perceptron(4)

#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         print(sess.run(out, feed_dict = {X : x}))
#         # print(sess.run(max_v, feed_dict = {in_value : 3.0}))
#     # tests here

# if __name__ == '__main__':
#     main()


# def main():
#     mnist = input_data.read_data_sets('data/mnist', one_hot=True)
#     X = qfns.input_placeholder()
#     Y = qfns.target_placeholder()

#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)

# if __name__ == '__main__':
#     main()
import tensorflow as tf
import numpy as np

from process_data import fetch_data
import math
import pickle

def random_mini_batches(X, Y, mini_batch_size, seed):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_H,n_W,n_C,n_y):

    X=tf.placeholder(tf.float32,shape=(None,n_H,n_W,n_C))
    Y=tf.placeholder(tf.float32,shape=(None,n_y))

    return X,Y

def initialize_paramters():
    parameters = {}


    W1= tf.get_variable("W1",[4,4,3,16],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W2 = tf.get_variable("W2", [4, 4, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W3 = tf.get_variable("W3", [4, 4, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=1))

    parameters['W1'] = W1
    parameters['W2'] = W2
    parameters['W3'] = W3
    return parameters

def forward_propogation(X,parameters):
    W1= parameters['W1']
    W2=parameters['W2']
    W3=parameters['W3']

    Z1= tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
    A1=tf.nn.relu(Z1)
    P1=tf.nn.max_pool(A1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    N1=tf.layers.batch_normalization(P1)

    Z2 = tf.nn.conv2d(N1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    N2 = tf.layers.batch_normalization(P2)

    Z3 = tf.nn.conv2d(N2, W3, strides=[1, 1, 1, 1], padding="SAME")
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    N3 = tf.layers.batch_normalization(P3)

    F1=tf.contrib.layers.flatten(N3)
    Z4= tf.contrib.layers.fully_connected(F1,10,activation_fn=None)

    return Z4

def compute_cost(Z,Y):

    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z,labels=Y))

    return cost



def model(X_train, Y_train, X_test, Y_test,learning_rate,num_epochs, minibatch_size):

    m,n_H,n_W,n_C= X_train.shape
    m,n_Y= Y_train.shape
    costs=[]
    seed = 3
    X,Y=create_placeholders(n_H,n_W,n_C,n_Y)

    parameters=initialize_paramters()

    Z=forward_propogation(X,parameters)

    cost=compute_cost(Z,Y)

    optimizer= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost=0
            num_minibatches=int(m/minibatch_size)

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _, temp_cost = sess.run(
                    fetches=[optimizer, cost],
                    feed_dict={X: minibatch_X,
                               Y: minibatch_Y}
                )

                minibatch_cost += temp_cost / num_minibatches

            if epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        parameters = sess.run(parameters)

        correct_prediction = tf.equal(tf.argmax(Z,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters



X_train,X_test,Y_train,Y_test=fetch_data()

parameters=model(X_train,Y_train,X_test,Y_test,0.001,60,64)

with open("mySavedDict.txt", "wb") as myFile:
    pickle.dump(parameters, myFile)




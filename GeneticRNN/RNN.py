import numpy as np
import shelve
import tensorflow as tf
import math
import os
from interface import implements, Interface
from INetwork import INetwork
from Genome import Genome


class RNN(implements(INetwork)):

    def __init__(self, my_shelve, genomes):
        self.n_hidden = 32
        self.n_classes = 6
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_iters = 300
        self.batch_size = 1500,
        self.display_iter = 30000

        for genome in genomes:
            setattr(self, genome._genName, genome._value)

    def RunAndAccuracy(self):
        # preparing data
        # 1-read datda
        my_shelve = shelve.open(self.my_shelve)
        features_test = my_shelve['features_test']
        features = my_shelve['features']
        y_test = my_shelve['labels_test']
        y_tr = my_shelve['labels_train']
        X_tr = my_shelve['data_train']
        X_test = my_shelve['data_test']
        my_shelve.close()

        # 7352 training series (with 50% overlap between each serie)
        self.training_data_count = len(self.X_train)
        self.test_data_count = len(self.X_test)  # 2947 testing series
        self.n_steps = len(self.X_train[0])  # 128 timesteps per series
        # 9 input parameters per timestep
        self.n_input = len(self.X_train[0][0])
        self.training_iters = self.training_data_count * 300

        x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_input])
        y = tf.placeholder(tf.float32, [None, self.n_classes])

        # Graph weights
        weights = {
            'hidden': tf.Variable(tf.random_normal([self.n_input,
                                                    self.n_hidden])),
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes], mean=1.0))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        pred = self.LSTM_RNN(x, weights, biases)

        # Loss, optimizer and evaluation
        l2 = self.lambda_loss_amount * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        )  # L2 loss prevents this overkill neural network to overfit the data
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=pred)) + l2  # Softmax loss
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(cost)  # Adam Optimizer

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        test_losses = []
        test_accuracies = []
        train_losses = []
        train_accuracies = []

        # Launch the graph
        sess = tf.InteractiveSession(
            config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 1
        while step * self.batch_size <= self.training_iters:
            batch_xs = self.extract_batch_size(
                self.X_train, step, self.batch_size)
            batch_ys = self.one_hot(self.extract_batch_size(
                self.y_train, step, self.batch_size))

            # Fit training using batch data
            _, loss, acc = sess.run(
                [optimizer, cost, accuracy],
                feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }
            )
            train_losses.append(loss)
            train_accuracies.append(acc)

            # Evaluate network only at some steps for faster training:
            if (step*self.batch_size % self.display_iter == 0) or (step == 1) or (step * self.batch_size > self.training_iters):

                # To not spam console, show training accuracy/loss in this "if"
                print("Training iter #" + str(step*self.batch_size) +
                      ":   Batch Loss = " + "{:.6f}".format(loss) +
                      ", Accuracy = {}".format(acc))

                # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
                loss, acc = sess.run(
                    [cost, accuracy],
                    feed_dict={
                        x: self.X_test,
                        y: self.one_hot(self.y_test)
                    }
                )
                test_losses.append(loss)
                test_accuracies.append(acc)
                print("PERFORMANCE ON TEST SET: " +
                      "Batch Loss = {}".format(loss) +
                      ", Accuracy = {}".format(acc))

            step += 1

        print("Optimization Finished!")

        # Accuracy for test data

        one_hot_predictions, accuracy, final_loss = sess.run(
            [pred, accuracy, cost],
            feed_dict={
                x: self.X_test,
                y: self.one_hot(self.y_test)
            }
        )

        test_losses.append(final_loss)
        test_accuracies.append(accuracy)

        print("FINAL RESULT: " +
              "Batch Loss = {}".format(final_loss) +
              ", Accuracy = {}".format(accuracy))

    def LSTM_RNN(self, _X, _weights, _biases):
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [-1, self.n_input])
        # new shape: (n_steps*batch_size, n_input)

        _X = tf.nn.relu(
            tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
        _X = tf.split(_X, self.n_steps, 0)
        # new shape: n_steps * (batch_size, n_hidden)
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(
            self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(
            self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        # Get LSTM cell output
        outputs, states = tf.contrib.rnn.static_rnn(
            lstm_cells, _X, dtype=tf.float32)
        lstm_last_output = outputs[-1]

        # Linear activation
        return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

    def extract_batch_size(self, _train, step, batch_size):
        shape = list(_train.shape)
        shape[0] = batch_size
        batch_s = np.empty(shape)
        for i in range(batch_size):
            # Loop index
            index = ((step-1)*batch_size + i) % len(_train)
            batch_s[i] = _train[index]
        return batch_s

    def one_hot(self, y_):
        y_ = y_.reshape(len(y_))
        return np.eye(self.n_classes)[np.array(y_, dtype=np.int32)]

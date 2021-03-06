import numpy as np
import shelve
import tensorflow as tf
import math
import os
import Utility as ut
from sklearn.metrics import classification_report


class CNN:
    def __init__(self, my_shelve, segment_size=128, n_filters=196,
                 n_channels=6, epochs=200, batch_size=200, learning_rate=5e-4,
                 dropout_rate=0.05, eval_iter=10, filters_size=16, n_classes=6,
                 IncludeFeat=0):
      # I'd like to use layers as parameters, flag to use features or not
        self.n_hidden = 1024
        self.l2_reg = 5e-4
        self.segment_size = segment_size
        self.n_channels = n_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.eval_iter = eval_iter
        self.n_filters = n_filters
        self.filters_size = filters_size
        self.n_classes = n_classes  # it isn't valid to change it
        self.my_shelve = my_shelve
        # self.n_layers = n_layers
        self.IncludeFeat = IncludeFeat

    def RunAndAccuracy(self):
        # preparing data
        # 1-read datda
        my_shelve = shelve.open(self.my_shelve)
        features_test = my_shelve['features_test']
        features = my_shelve['features']
        labels_test = my_shelve['labels_test']
        labels_train = my_shelve['labels_train']
        data_train = my_shelve['data_train']
        data_test = my_shelve['data_test']
        my_shelve.close()

        # for i in range(self.n_channels):
        #     x = data_train[i * self.segment_size: (i + 1) * self.segment_size, :]
        #     data_train[i * self.segment_size: (i + 1) * self.segment_size, :] = ut.norm(x)
        #     x = data_test[i * self.segment_size: (i + 1) * self.segment_size, :]
        #     data_test[i * self.segment_size: (i + 1) * self.segment_size,:] = ut.norm(x)

        # 2 Reshape data
        data_train = np.reshape(
            data_train, [-1, self.segment_size, self.n_channels])
        data_test = np.reshape(
            data_test, [-1, self.segment_size, self.n_channels])
        labels_train = np.reshape(labels_train, [-1, self.n_classes])
        labels_test = np.reshape(labels_test, [-1, self.n_classes])
        labels_test_unary = np.argmax(labels_test, axis=1)

        # 3 collect size
        train_size = data_train.shape[0]
        test_size = data_test.shape[0]
        num_features = features.shape[1]

        W_conv1 = self.weight_variable(
            [1, self.filters_size, self.n_channels, self.n_filters], stddev=0.01)
        b_conv1 = self.bias_variable([self.n_filters])

        x = tf.placeholder(
            tf.float32, [None, self.segment_size * self.n_channels])
        x_image = tf.reshape(x, [-1, 1, self.segment_size, self.n_channels])

        h_conv1 = tf.nn.relu(self.conv1d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_1x4(h_conv1)

        # Augmenting data with statistical features

        flat_size = int(math.ceil(float(self.segment_size)/4)) * self.n_filters

        h_feat = tf.placeholder(tf.float32, [None, num_features])
        h_flat = tf.reshape(h_pool1, [-1, flat_size])
        h_hidden = tf.concat(axis=1, values=[h_flat, h_feat])
        flat_size += num_features

        # Fully connected layer with Dropout

        W_fc1 = self.weight_variable([flat_size, self.n_hidden], stddev=0.01)
        b_fc1 = self.bias_variable([self.n_hidden])

        h_fc1 = tf.nn.relu(tf.matmul(h_hidden, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Softmax layer

        W_softmax = self.weight_variable(
            [self.n_hidden, self.n_classes], stddev=0.01)
        b_softmax = self.bias_variable([self.n_classes])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_softmax) + b_softmax)
        y_ = tf.placeholder(tf.float32, [None, self.n_classes])

        # Cross entropy loss function and L2 regularization term

        cross_entropy = - \
            tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
        cross_entropy += self.l2_reg * \
            (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1))

        # Training step

        train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Run Tensorflow session

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        # Train CNN
        print("Training CNN... ")

        max_accuracy = 0.0
        data_test = np.reshape(
            data_test, [data_test.shape[0], self.segment_size * self.n_channels])
        if (os.path.exists('checkpoints1-cnn') == False):
            os.mkdir('checkpoints1-cnn')
        saver = tf.train.Saver()
        for i in range(100000):

            idx_train = np.random.randint(0, train_size, self.batch_size)

            xt = np.reshape(data_train[idx_train], [
                            self.batch_size, self.segment_size * self.n_channels])
            yt = np.reshape(labels_train[idx_train], [
                            self.batch_size, self.n_classes])
            ft = np.reshape(features[idx_train], [
                            self.batch_size, num_features])

            sess.run(train_step, feed_dict={
                     x: xt, y_: yt, h_feat: ft, keep_prob: self.dropout_rate})

            if i % self.eval_iter == 0:

                train_accuracy, train_entropy, y_pred = sess.run([accuracy, cross_entropy, y_conv],
                                                                 feed_dict={x: data_test, y_: labels_test, h_feat: features_test, keep_prob: 1})

                print("step %d, entropy %g" % (i, train_entropy))
                print("step %d, max accuracy %g, accuracy %g" %
                      (i, max_accuracy, train_accuracy))
                print(classification_report(labels_test_unary,
                                            np.argmax(y_pred, axis=1), digits=4))

                if max_accuracy < train_accuracy:
                    max_accuracy = train_accuracy
        saver.save(sess, "checkpoints1-cnn/har.ckpt")

    #    # prepare layers ANOTHER TECHNIQUE
    #     graph = tf.Graph()
    #     with graph.as_default():
    #         inputs_ = tf.placeholder(
    #             tf.float32, [None, self.segment_size, self.n_channels], name='inputs')
    #         labels_ = tf.placeholder(
    #             tf.float32, [None, self.n_classes], name='labels')
    #         keep_prob_ = tf.placeholder(tf.float32, name='keep')
    #         learning_rate_ = tf.placeholder(tf.float32, name='Learning_rate')
    #         h_feat = tf.placeholder(
    #             tf.float32, [None, num_features], name='features')

    #     # build conv and pool layers
    #     with graph.as_default():
    #         # (batch, 128, 9) --> (batch, 64, 18)
    #         conv1 = tf.layers.conv1d(inputs=inputs_, filters=18,
    #                                  kernel_size=2, strides=1,
    #                                  padding='same', activation=tf.nn.relu)
    #         max_pool_1 = tf.layers.max_pooling1d(
    #             inputs=conv1, pool_size=2, strides=2, padding='same')

    #         # (batch, 64, 18) --> (batch, 32, 36)
    #         conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36,
    #                                  kernel_size=2, strides=1,
    #                                  padding='same', activation=tf.nn.relu)
    #         max_pool_2 = tf.layers.max_pooling1d(
    #             inputs=conv2, pool_size=2, strides=2, padding='same')

    #         # (batch, 32, 36) --> (batch, 16, 72)
    #         conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72,
    #                                  kernel_size=2, strides=1,
    #                                  padding='same', activation=tf.nn.relu)
    #         max_pool_3 = tf.layers.max_pooling1d(
    #             inputs=conv3, pool_size=2, strides=2, padding='same')

    #         # (batch, 16, 72) --> (batch, 8, 144)
    #         conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144,
    #                                  kernel_size=2, strides=1,
    #                                  padding='same', activation=tf.nn.relu)
    #         max_pool_4 = tf.layers.max_pooling1d(
    #             inputs=conv4, pool_size=2, strides=2, padding='same')

    #     # flat and add features
    #     with graph.as_default():
    #         # connect feature + flat as one layer
    #         shape = max_pool_4.get_shape().as_list()
    #         flat_size = shape[1]*shape[2]
    #         h_flat = tf.reshape(max_pool_4, (-1, flat_size))
    #         # feat_flat = tf.concat(axis=1, values=[h_flat, h_feat])
    #         # flat_size += num_features
    #         # add dropout
    #         # if self.IncludeFeat == 1:
    #         #     flat = tf.nn.dropout(feat_flat, keep_prob=keep_prob_)
    #         # else:
    #         flat = tf.nn.dropout(h_flat, keep_prob=keep_prob_)

    #         logits = tf.layers.dense(flat, self.n_classes)

    #         # Cost function and optimizer
    #         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #             logits=logits, labels=labels_))
    #         optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

    #     # Accuracy
    #         correct_pred = tf.equal(
    #             tf.argmax(logits, 1), tf.argmax(labels_, 1))
    #         accuracy = tf.reduce_mean(
    #             tf.cast(correct_pred, tf.float32), name='accuracy')

    #     # Train the network
    #     train_acc = []
    #     train_loss = []
    #     if (os.path.exists('checkpoints-cnn') == False):
    #         os.mkdir('checkpoints-cnn')

    #     with graph.as_default():
    #         saver = tf.train.Saver()
    #         # run tensorflow session
    #     with tf.Session(graph=graph) as sess:
    #         sess.run(tf.global_variables_initializer())
    #         iteration = 1
    #         print("Training CNN... ")
    #         # Loop over ephoches
    #         for e in range(self.epochs):
    #             # Loop over batches
    #             for x, y in self.get_batches(X_tr, y_tr, self.batch_size):
    #                 # Feed dictionary
    #                 feed = {inputs_: x, labels_: y, keep_prob_: self.keep_prob,
    #                         learning_rate_: self.learning_rate}
    #                 # Loss
    #                 sess.run(optimizer, feed_dict=feed)
    #         saver.save(sess, "checkpoints-cnn/har.ckpt")
    #         test_acc = []
    #         for x_t, y_t in self.get_batches(X_test, y_test, self.batch_size):
    #             feed = {inputs_: x_t,
    #                     labels_: y_t,
    #                     keep_prob_: 1}
    #             batch_acc = sess.run(accuracy, feed_dict=feed)
    #             test_acc.append(batch_acc)
    #         print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

    #    # with tf.Session(graph=graph) as sess:
    #         # Restore
    #        #  saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))

    def get_batches(self, X, y, batch_size=100):
        """ Return a generator for batches """
        n_batches = len(X) // batch_size
        X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]
        # Loop over batches and yield
        for b in range(0, len(X), batch_size):
            yield X[b:b+batch_size], y[b:b+batch_size]

    def weight_variable(self, shape, stddev):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv1d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_1x4(self, x):
        return tf.nn.max_pool(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME')


import numpy as np
import shelve
import tensorflow as tf
import math
import os
import utilities as ut
from tensorflow.saved_model import tag_constants


class CNN:
    def __init__(self, my_shelve, segment_size=128, n_filters=196,
                 n_channels=6, epochs=200, batch_size=200, learning_rate=5e-4,
                 keep_prob=0.05, eval_iter=10, filters_size=16, n_classes=6,IncludeFeat):
      # I'd like to use layers as parameters, flag to use features or not

        self.segment_size = segment_size
        self.n_channels = n_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.eval_iter = eval_iter
        self.n_filters = n_filters
        self.filters_size = filters_size
        self.n_classes = n_classes  # it isn't valid to change it
        self.my_shelve = my_shelve
        # self.n_layers = n_layers
        self.IncludeFeat = IncludeFeat
        self.num_features = 561

    def RunAndTraining(self):
        # preparing data
        # 1-read datda
        my_shelve = shelve.open(self.my_shelve)
        X_tr = my_shelve['data_train']
        # X_vld = my_shelve['data_vld']
        X_test = my_shelve['data_test']
        y_tr = my_shelve['labels_train']
        y_test = my_shelve['labels_test']
        # y_vld = my_shelve['labels_vld']
        features_train = my_shelve['features_train']
        features_test = my_shelve['features_test']
        my_shelve.close()

        graph = tf.Graph()
        # Construct placeholders
        with graph.as_default():
            inputs_ = tf.placeholder(tf.float32, [None, self.segment_size,
                                                  self.n_channels],
                                     name='inputs')
            h_feat = tf.placeholder(
                tf.float32, [None, self.num_features], name='features')
            labels_ = tf.placeholder(
                tf.float32, [None, self.n_classes], name='labels')
            keep_prob_ = tf.placeholder(tf.float32, name='keep')
            learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

        # build conv and pool layers
        with graph.as_default():
            # (batch, 128, 9) --> (batch, 64, 18)
            conv1 = tf.layers.conv1d(inputs=inputs_, filters=18,
                                     kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_1 = tf.layers.max_pooling1d(
                inputs=conv1, pool_size=2, strides=2, padding='same')

            # (batch, 64, 18) --> (batch, 32, 36)
            conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36,
                                     kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_2 = tf.layers.max_pooling1d(
                inputs=conv2, pool_size=2, strides=2, padding='same')

            # (batch, 32, 36) --> (batch, 16, 72)
            conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72,
                                     kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_3 = tf.layers.max_pooling1d(
                inputs=conv3, pool_size=2, strides=2, padding='same')

            # (batch, 16, 72) --> (batch, 8, 144)
            conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144,
                                     kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_4 = tf.layers.max_pooling1d(
                inputs=conv4, pool_size=2, strides=2, padding='same')

        # flat and add features
        with graph.as_default():
            # connect feature + flat as one layer
            shape = max_pool_4.get_shape().as_list()
            flat_size = shape[1]*shape[2]
            h_flat = tf.reshape(max_pool_4, (-1, flat_size))
            feat_flat = tf.concat(axis=1, values=[h_flat, h_feat])
            flat_size += self.num_features
            # add dropout
            if self.IncludeFeat == 1:
                flat = tf.nn.dropout(feat_flat, keep_prob=keep_prob_)
            else:
                flat = tf.nn.dropout(h_flat, keep_prob=keep_prob_)

            logits = tf.layers.dense(flat, self.n_classes)

            # Cost function and optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels_))
            optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

        # Accuracy
            correct_pred = tf.equal(
                tf.argmax(logits, 1), tf.argmax(labels_, 1))
            accuracy = tf.reduce_mean(
                tf.cast(correct_pred, tf.float32), name='accuracy')

        # Train the network
        # validation_acc = []
        # validation_loss = []
        train_acc = []
        train_loss = []
        if (os.path.exists('checkpoints-cnn') == False):
            os.mkdir('checkpoints-cnn')

        with graph.as_default():
            saver = tf.train.Saver()
            # run tensorflow session
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            print("Training CNN... ")
            # Loop over ephoches
            for e in range(self.epochs):
                # Loop over batches
                for x, y, ft in ut.get_batches(X_tr, y_tr, features_train, self.batch_size):
                    # Feed dictionary
                    feed = {inputs_: x, labels_: y, h_feat: ft, keep_prob_: 0.5,
                            learning_rate_: self.learning_rate}
                    # Loss
                    loss, _, acc = sess.run(
                        [cost, optimizer, accuracy], feed_dict=feed)
                    train_acc.append(acc)
                    train_loss.append(loss)
                    if (iteration % 10 == 0):
                        print("Epoch: {}/{}".format(e, self.epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:.6f}".format(acc))

                    # # Compute validation loss at every 10 iterations
                    # if (iteration % 10 == 0):
                    #     val_acc_ = []
                    #     val_loss_ = []

                    #     for x_v, y_v in ut.get_batches(X_vld, y_vld, self.batch_size):
                    #         # Feed
                    #         feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1.0}

                    #         # Loss
                    #         loss_v, acc_v = sess.run([cost, accuracy], feed_dict=feed)
                    #         val_acc_.append(acc_v)
                    #         val_loss_.append(loss_v)

                    #     # Print info
                    #     print("Epoch: {}/{}".format(e, self.epochs),
                    #         "Iteration: {:d}".format(iteration),
                    #         "Validation loss: {:6f}".format(np.mean(val_loss_)),
                    #         "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                    #     # Store
                    #     validation_acc.append(np.mean(val_acc_))
                    #     validation_loss.append(np.mean(val_loss_))

                    # Iterate
                    iteration += 1
            saver.save(sess, "checkpoints-cnn/har.ckpt")

    def Testing(self):
        my_shelve = shelve.open(self.my_shelve)
        X_test = my_shelve['data_test']
        y_test = my_shelve['labels_test']
        features_test = my_shelve['features_test']
        my_shelve.close()

        test_acc = []
        graph = tf.get_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("checkpoints-cnn/har.ckpt.meta")
            saver.restore(sess, tf.train.latest_checkpoint("checkpoints-cnn"))
            inputs_ = graph.get_tensor_by_name('inputs:0')
            labels_ = graph.get_tensor_by_name('labels:0')
            keep_prob_ = graph.get_tensor_by_name('keep:0')
            h_feat = graph.get_tensor_by_name('features:0')
            accuracy = graph.get_tensor_by_name('accuracy:0')
            # Restore
            for x_t, y_t, ft_t in ut.get_batches(X_test, y_test, features_test, self.batch_size):
                feed = {inputs_: x_t,
                        labels_: y_t,
                        h_feat: ft_t,
                        keep_prob_: 1}
                batch_acc = sess.run(accuracy, feed_dict=feed)
                test_acc.append(batch_acc)
        print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

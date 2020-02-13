import numpy as np
import shelve
import tensorflow as tf
import math
import os
from interface import implements, Interface
from INetwork import INetwork
from Genome import Genome
class CNN(implements(INetwork)):
    def __init__(self,my_shelve, genomes):
#         segment_size=128, n_filters=196,n_channels = 6,
#         epochs = 200,batch_size = 200,learning_rate = 5e-4,
#         keep_prob = 0.05,eval_iter = 10,filters_size = 16,n_classes = 6
  # I'd like to use layers as parameters, flag to use features or not

        self.segment_size = 128
        self.n_channels = 6
        self.epochs = 200
        self.batch_size = 200
        self.learning_rate = 5e-4
        self.keep_prob = 0.05
        self.eval_iter = 10
        self.n_filters = 196
        self.filters_size = 16
        self.n_classes = 6 #it isn't valid to change it
        self.my_shelve=my_shelve
        self.n_layers=1
        self.IncludeFeat=0

        for genome in genomes:
            setattr(self,genome._genName,genome._value)


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

        # 2 Reshape data
        X_tr = np.reshape(X_tr, [-1, self.segment_size, self.n_channels])
        X_test = np.reshape(X_test, [-1, self.segment_size, self.n_channels])
        y_tr = np.reshape(y_tr, [-1, self.n_classes])
        y_test = np.reshape(y_test, [-1, self.n_classes])

        # 3 collect size
        train_size = X_tr.shape[0]
        test_size = X_test.shape[0]
        num_features = features.shape[1]

        # prepare layers
        graph = tf.Graph()
        with graph.as_default():
            inputs_ = tf.placeholder(
                tf.float32, [None, self.segment_size, self.n_channels], name='inputs')
            labels_ = tf.placeholder(
                tf.float32, [None, self.n_classes], name='labels')
            keep_prob_ = tf.placeholder(tf.float32, name='keep')
            learning_rate_ = tf.placeholder(tf.float32, name='Learning_rate')
            h_feat = tf.placeholder(
                tf.float32, [None, num_features], name='features')

        # build conv and pool layers
        with graph.as_default():
            self.conv0=tf.layers.conv1d(inputs=inputs_,filters=self.n_filters,kernel_size=self.filters_size,strides=1,padding='same',activation=tf.nn.relu)
            self.max_pool0=tf.layers.max_pooling1d(inputs=self.conv0,pool_size=4,strides=4,padding='same')
            i=1
            namemax_Pool="max_pool0"
            while(i<self.n_layers):
                nameConv="conv"+str(i)
                val=tf.layers.conv1d(inputs=getattr(self,"max_pool"+str(i-1)),filters=self.n_filters,kernel_size=self.filters_size,strides=1,padding='same',activation=tf.nn.relu)
                setattr(self,nameConv,val)
                namemax_Pool="max_pool"+str(i)
                val=tf.layers.max_pooling1d(inputs=getattr(self,nameConv),pool_size=4,strides=4,padding='same')
                setattr(self,namemax_Pool,val)
                i+=1

        # flat and add features
        with graph.as_default():
            # connect feature + flat as one layer
            lastmax_pool=getattr(self,namemax_Pool)
            shape = lastmax_pool.get_shape().as_list()
            flat_size=shape[1]*shape[2]
            h_flat = tf.reshape(lastmax_pool, (-1, flat_size))
            feat_flat = tf.concat(axis=1, values=[h_flat, h_feat])
            flat_size += num_features
            # add dropout
            if self.IncludeFeat==1:
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
        # if (os.path.exists('checkpoints-cnn') == False):
        #     os.mkdir('checkpoints-cnn')

        #with graph.as_default():
            # saver = tf.train.Saver()
            # run tensorflow session
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            print("Training CNN... ")
            for e in range(self.epochs):
                # Loop over batches
                for x, y, ft in self.get_batches(X_tr, y_tr, features, self.batch_size):
                    # Feed dictionary
                    feed = {inputs_: x, labels_: y, h_feat: ft,
                            keep_prob_: self.keep_prob, learning_rate_: self.learning_rate}
                    # Loss
                    sess.run(optimizer, feed_dict=feed)
                # if (e % self.eval_iter == 0):
                #     feed = {inputs_: X_test, labels_: y_test, h_feat: features_test,
                #             keep_prob_: 1, learning_rate_: self.learning_rate}
                #     train_accuracy = sess.run(accuracy, feed_dict=feed)
                #     print("step %d, accuracy %g" % (e, train_accuracy))
             #saver.save(sess,"checkpoints-cnn/har.ckpt")
            test_acc = []
            for x_t, y_t, ft_tst in self.get_batches(X_test, y_test, features_test, self.batch_size):
                feed = {inputs_: x_t,
                        labels_: y_t,
                        h_feat: ft_tst,
                        keep_prob_: self.keep_prob}
                batch_acc = sess.run(accuracy, feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.6f}".format(np.mean(test_acc)))
            acc=np.mean(test_acc)
            return format(acc)

       # with tf.Session(graph=graph) as sess:
            # Restore
           #  saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))
            


    def get_batches(self, X, y, ft, batch_size=100):
        """ Return a generator for batches """
        n_batches = len(X) // batch_size
        X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]
        # Loop over batches and yield
        for b in range(0, len(X), batch_size):
            yield X[b:b+batch_size], y[b:b+batch_size], ft[b:b+batch_size]

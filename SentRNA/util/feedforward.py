import numpy as np
import tensorflow as tf
import os
import random
from compute_mfe import *
from featurize_util import *


class TensorflowClassifierModel():
    def __init__(self, layer_sizes, logdir='test',
                 weight_init_stddevs=[.02], bias_init_consts=[1.], learning_rate=1e-3,**kwargs):
        # Save hyperparameters
        self.layer_sizes = layer_sizes
        self.weight_init_stddevs = weight_init_stddevs
        self.bias_init_consts = bias_init_consts
        self.learning_rate = learning_rate

        if logdir is not None:
            if not os.path.exists(logdir):
                os.makedirs(logdir)
        else:
            logdir = tempfile.mkdtemp()
        self.logdir = logdir

        self.graph = tf.Graph()
        self.output = self.build()

    def full_forward(self, n_layers, x, weights, biases):
        layer = tf.add(tf.matmul(x, weights[0]), biases[0])
        layer = tf.nn.relu(layer)
        for i in range(1, n_layers - 1):
            layer = tf.add(tf.matmul(layer, weights[i]), biases[i])
            layer = tf.nn.relu(layer)
        return tf.matmul(layer, weights[n_layers - 1]) + biases[n_layers - 1]

    def add_label_placeholders(self):
        return tf.placeholder("float", [None, self.layer_sizes[-1]], name='labels')

    def build(self):
        with self.graph.as_default():
            weights = []
            biases = []
            for i in range(len(self.layer_sizes) - 1):
                weights.append(tf.Variable(tf.random_normal([self.layer_sizes[i], self.layer_sizes[
                               i + 1]],  stddev=self.weight_init_stddevs, name='W%d' % (i + 1))))
                biases.append(tf.Variable(
                    tf.zeros([1, self.layer_sizes[i + 1]]), name='b%d' % (i + 1)))
            self.Weights = weights
            self.Biases = biases
            self.x = tf.placeholder(
                "float", [None, self.layer_sizes[0]], name='x')
            self.labels = self.add_label_placeholders()
            self.rewards = tf.placeholder("float", name='rewards')
            # Forward pass
            output = self.full_forward(
                len(self.layer_sizes) - 1, self.x, weights, biases)
            self.loss = self.cost(output, self.labels, self.rewards) #+ model_ops.weight_decay(self.penalty_type, self.penalty)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            return output

    def cost(self, output, labels, rewards):
        return tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels), rewards))

    def construct_feed_dict(self, x, labels, rewards):
        return {self.x: np.array(x), self.labels: np.array(labels), self.rewards: np.array(rewards)}

    def fit(self, dataset, loss_thresh, nb_epochs, save_path, checkpoint=None, **kwargs):
        with self.graph.as_default():  
            sess = tf.Session()      
            train_op = self.train_op
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if checkpoint:
                saver.restore(sess, checkpoint)
            # Save initial model
            X, y, w, _ = dataset
            epoch = 0
            while epoch < nb_epochs:
                print epoch
                epoch += 1
                _, loss, output = sess.run([train_op, self.loss, self.output], feed_dict=self.construct_feed_dict(X, y, w))
                print loss
                #if loss < loss_thresh:
                #    saver.save(sess, self._save_path, global_step=epoch)
                #    break
            saver.save(sess, os.path.join(self.logdir, save_path), global_step=epoch + 1)

    def evaluate(self, dot_bracket, seq, fixed_bases, layer_sizes, MI_features_list, checkpoint, refine, MI_tolerance, renderer, **kwargs):
        with self.graph.as_default():
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess, checkpoint)
            rec = []
            to_solve = 'n' * len(dot_bracket)
            for fixed_base in fixed_bases:
                to_solve = insert_base(to_solve, seq[fixed_base], fixed_base)
            open_positions = []
            for i in range(len(to_solve)):
                if to_solve[i] == 'n':
                    open_positions.append(i)
            if refine:
                to_solve = seq
            for i in open_positions:
                inputs, label = prepare_single_base_environment(dot_bracket, to_solve, i, MI_features_list, MI_tolerance, renderer)
                rec.append(inputs)
                X, y, w = [inputs], [label], [1]
                output = sess.run(self.output, feed_dict = self.construct_feed_dict(X, y, w))[0]
                pred_base = output_to_base(output)
                to_solve = insert_base(to_solve, pred_base, i)
            return to_solve, rec

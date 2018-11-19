import tensorflow as tf
import numpy as np
import collections
from config import Config
from common import get_initializer, get_activation

State = collections.namedtuple('State', ['fix_combinations', 'cur_combination'])


class Actor:
    '''
        num_fix_combinations
        num_cur_fields
        fix_combinations
        cur_combination
        action: the corresponding action for every state, the i means select the ith available action
        target
        fix_encoded
        cur_encoded
        fix_combined
        chooser_input
        logits
        action_probs
        loss
        train_op
    '''
    def __init__(self, graph, sess, optimizer):
        self.graph = graph
        self.sess = sess
        self.optimizer = optimizer

        self.num_fix_combinations = None  # []
        self.num_cur_fields = None  # []
        self.fix_combinations = None  # batch * num_fix * num_fields
        self.cur_combination = None  # batch * num_fields
        self.action = None  # batch
        self.target = None  # batch
        self.fix_encoded = None  # batch * num_fix * encode_dim
        self.cur_encoded = None  # batch * encode_dim
        self.fix_combined = None  # batch * encode_dim
        self.chooser_input = None  # batch * 2 encode_dim
        self.logits = None  # batch * num_fields
        self.action_probs = None  # batch * num_fields
        self.loss = None  # []
        self.train_op = None  # operation

        with graph.as_default():
            self.define_inputs()

            self.define_encoder(Config.encoder_dim)

            self.define_combinator()

            self.chooser_input = tf.concat([self.fix_combined, self.cur_encoded], axis=1)
            self.define_chooser([('full', 1024), ('act', 'relu'), ('full', 128), ('act', 'relu'), ('full', Config.num_fields)])

            self.define_loss_and_train()

    def define_inputs(self):
        with tf.variable_scope("inputs"):
            self.num_fix_combinations = tf.placeholder(dtype=tf.int32, shape=[], name="num_fix_combinations")
            self.num_cur_fields = tf.placeholder(dtype=tf.int32, shape=[], name="num_cur_fields")
            self.fix_combinations = tf.cast(tf.placeholder(dtype=tf.int32, shape=[None, None, Config.num_fields]),
                                            name="fix_combinations", dtype=tf.float32)
            self.cur_combination = tf.cast(tf.placeholder(dtype=tf.int32, shape=[None, Config.num_fields]), name="cur_combination",
                                           dtype=tf.float32)
            self.action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
            self.target = tf.placeholder(dtype=tf.float32, shape=[None], name="target")

    def define_encoder(self, encode_dim):
        with tf.variable_scope("encoder"):
            self.encoder_w = w = tf.get_variable("w", shape=[Config.num_fields, encode_dim], dtype=tf.float32,
                                                 initializer=get_initializer("xavier"))
            self.fix_encoded = tf.sigmoid(tf.tensordot(self.fix_combinations, w, 1), name="fix_encoded")  # batch * fix_num * encode_dim
            self.cur_encoded = tf.sigmoid(tf.tensordot(self.cur_combination, w, 1), name="cur_encoded")  # batch * encode_dim

    def define_combinator(self):
        with tf.variable_scope("combinator"):
            self.fix_combined = tf.reduce_mean(self.fix_encoded, axis=1, name="fix_combined")  # batch * encode_dim

    def define_chooser(self, layers):
        with tf.variable_scope("chooser"):
            layer_index = 0
            cur_layer = self.chooser_input
            cur_dim = self.chooser_input.get_shape().as_list()[-1]
            for layer_type, layer_param in layers:
                if layer_type == 'full':
                    new_dim = layer_param
                    with tf.variable_scope("layer_{}".format(layer_index)):
                        w = tf.get_variable("w", shape=[cur_dim, new_dim], dtype=tf.float32, initializer=get_initializer("xavier"))
                        b = tf.get_variable("b", shape=[1, new_dim], dtype=tf.float32, initializer=get_initializer(0.0))
                        cur_layer = tf.matmul(cur_layer, w) + b
                    layer_index += 1
                    cur_dim = new_dim
                elif layer_type == 'act':
                    cur_layer = get_activation(layer_param)(cur_layer)
                else:
                    raise ValueError
            self.logits = tf.reshape(tf.boolean_mask(cur_layer, tf.equal(self.cur_combination, 0)),
                                     shape=[-1, Config.num_fields - self.num_cur_fields], name="logits")
            self.action_probs = tf.nn.softmax(self.logits)

    def define_loss_and_train(self):
        self.loss = tf.reduce_mean(self.target * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action, logits=self.logits),
                                   name="loss")
        self.train_op = self.optimizer.minimize(self.loss)

    def watch(self, fetches, feed_dict):
        return self.sess.run(fetches=fetches, feed_dict=feed_dict)

    def predict(self, state):
        """
        @:return probability distribution of actions
        """
        assert isinstance(state, State)
        fix_combs = state.fix_combinations[np.newaxis, :, :]
        cur_combs = state.cur_combination[np.newaxis, :]
        fetches = [self.action_probs, self.logits]
        feed_dict = {self.num_cur_fields: np.sum(state.cur_combination), self.fix_combinations: fix_combs, self.cur_combination: cur_combs}
        action_probs, logits = self.sess.run(fetches=fetches, feed_dict=feed_dict)
#        debug_fetches = [self.fix_combinations, self.cur_combination, self.encoder_w, self.fix_combined, self.cur_encoded, self.logits, self.action_probs]
#        print("fix_combinations:\n{}\ncur_combination:\n{}\nencoder/w:\n{}\nfix_combined:\n{}\ncur_encoded:\n{}\nlogits:\n{}\naction_probs:\n{}".format(*self.watch(debug_fetches, feed_dict=feed_dict)))
        return action_probs, logits

    def update(self, fix_combs, cur_combs, target, action):
        """
        update the network and return the loss
        """
        fetches = [self.loss, self.train_op]
        feed_dict = {self.num_cur_fields: np.sum(cur_combs[0]),
                     self.fix_combinations: fix_combs,
                     self.cur_combination: cur_combs,
                     self.action: action,
                     self.target: target}
        loss, _ = self.sess.run(fetches=fetches, feed_dict=feed_dict)
        return loss

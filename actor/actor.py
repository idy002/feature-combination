import tensorflow as tf
import sys
from config import Config
from common import get_initializer, get_activation
import numpy as np


class Actor:
    def __init__(self):
        sys.stderr.write('Actor.__init__\n')
        self.fix_combinations = None  # batch * num_fix * num_fields
        self.cur_combination = None  # batch * 1 * num_fields
        self.fix_encoded = None  # batch * num_fix * encode_dim
        self.cur_encoded = None  # batch * encode_dim
        self.fix_combined = None  # batch * encode_dim
        self.chooser_input = None  # batch * 2 encode_dim
        self.logits = None  # batch * num_fields
        self.action = None  # batch * 1

        sys.stderr.write('start define inputs\n')
        self.define_inputs()

        sys.stderr.write('start define encoder\n')
        self.define_encoder(Config.encoder_dim)

        sys.stderr.write('start define combinator\n')
        self.define_combinator()

        sys.stderr.write('start define chooser\n')
        self.chooser_input = tf.concat([self.fix_combined, self.cur_encoded], axis=1)
        self.define_chooser([('full', 1024), ('act', 'relu'), ('full', 128), ('act', 'relu'), ('full', Config.num_fields)])

    def define_inputs(self):
        with tf.variable_scope("inputs"):
            self.fix_combinations = tf.cast(tf.placeholder(dtype=tf.int32, shape=[None, None, Config.num_fields]), name="fix_combinations", dtype=tf.float32)
            self.cur_combination = tf.cast(tf.placeholder(dtype=tf.int32, shape=[None, 1, Config.num_fields]), name="cur_combination", dtype=tf.float32)

    def define_encoder(self, encode_dim):
        with tf.variable_scope("encoder"):
            w = tf.get_variable("w", shape=[1, Config.num_fields, encode_dim], dtype=tf.float32, initializer=get_initializer("xavier"))
            self.fix_encoded = tf.sigmoid(tf.matmul(self.fix_combinations, w), name="fix_encoded")  # batch * fix_num * encode_dim
            self.cur_encoded = tf.sigmoid(tf.squeeze(tf.matmul(self.cur_combination, w), axis=[1]), name="cur_encoded")  # batch * encode_dim

    def define_combinator(self):
        with tf.variable_scope("combinator"):
            self.fix_combined = tf.reduce_mean(self.fix_encoded, axis=0, name="fix_combined")  # batch * encode_dim

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
            self.logits = cur_layer  # batch * num_fields
            assert self.logits.get_shape().as_list()[-1] == Config.num_fields, "Chooser need to get a vector with num_field length here"
            self.action = tf.multinomial(self.logits, 1, name="chosen", output_dtype=tf.int32)





import tensorflow as tf
from config import Config
from common import get_initializer, get_activation
import numpy as np


class Actor:
    def __init__(self):
        self.fix_combinations = None  # batch * num_fix * num_fields
        self.cur_combination = None  # batch * 1 * num_fields
        self.fix_encoded = None  # batch * num_fix * encode_dim
        self.cur_encoded = None  # batch * 1 * encode_dim
        self.fix_combined = None  # batch * encode_dim
        self.chooser_input = None  # batch * 2 encode_dim
        self.chosen = None  # batch * 1

        self.define_inputs()

        self.define_encoder(Config.encoder_dim)

        self.define_combinator()

        self.define_chooser([('full', 1024), ('act', 'relu'), ('full', 128), ('act', 'relu'), ('full', Config.num_fields)])

    def define_inputs(self):
        with tf.variable_scope("inputs"):
            self.fix_combinations = tf.placeholder(dtype=tf.int32, shape=[None, None, Config.num_fields], name="fix_combinations")
            self.cur_combination = tf.placeholder(dtype=tf.int32, shape=[None, 1, Config.num_fields], name="cur_combination")

    def define_encoder(self, encode_dim):
        with tf.variable_scope("encoder"):
            w = tf.get_variable("w", shape=[Config.num_fields, encode_dim], dtype=tf.float32, initializer=get_initializer("xavier"))
            self.fix_encoded = tf.sigmoid(tf.matmul(self.fix_combinations, w), name="fix_encoded")
            self.cur_encoded = tf.sigmoid(tf.matmul(self.cur_combination, w), name="cur_encoded")

    def define_combinator(self):
        with tf.variable_scope("combinator"):
            self.fix_combined = tf.reduce_mean(self.fix_encoded, axis=0, name="fix_combined")

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
            logits = cur_layer  # batch * num_fields
            assert logits.get_shape().as_list()[-1] == Config.num_fields, "Chooser need to get a vector with num_field length here"
            self.chosen = tf.multinomial(logits, 1, name="chosen", output_dtype=tf.int32)





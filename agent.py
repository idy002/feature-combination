import tensorflow as tf
from config import Config


class Agent:
    def __init__(self):
        self.X = tf.placeholder(tf.int32, (None,Config.num_fields), "X")
        self.hide1 = tf.layers.dense(
            inputs = tf.to_float(self.X, "ToFloat"),
            units = Config.num_fields * 10,
            activation = tf.nn.relu,
            kernel_initializer = tf.glorot_normal_initializer(),
            name = "hide1")
        self.hide2 = tf.layers.dense(
            inputs = self.hide1,
            units = Config.num_fields * 4,
            activation = tf.nn.relu,
            kernel_initializer = tf.glorot_normal_initializer(),
            name = "hide2")
        self.logits = tf.layers.dense(
            inputs = self.hide2,
            units = Config.num_fields,
            name = "logits")
        self.value = tf.layers.dense(
            inputs = self.hide2,
            units = 1,
            name = "value")
        self.action_prob = tf.nn.softmax(logits=self.logits)
        self.filter_logits = tf.where(tf.equal(self.X, 0), self.logits, self.logits - 1e9)
        self.action = tf.multinomial(self.filter_logits, 1)



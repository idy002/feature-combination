import tensorflow as tf
from config import Config


class Agent:
    def __init__(self):
        self.X = tf.placeholder(tf.int32, (None,Config.num_fields), "X")
        self.hide = tf.layers.dense(
            inputs = tf.to_float(self.X, "ToFloat"),
            units = Config.num_fields * 5,
            activation = tf.nn.relu,
            name = "hide")
        self.logits = tf.layers.dense(
            inputs = self.hide,
            units = Config.num_fields,
            name = "logits")
        self.value = tf.layers.dense(
            inputs = self.hide,
            units = 1,
            name = "value")
        self.action_prob = tf.nn.softmax(logits=self.logits)
        self.filter_logits = tf.where(tf.equal(self.X, 0), self.logits, self.logits - 1e9)
        self.action = tf.multinomial(self.filter_logits, 1)



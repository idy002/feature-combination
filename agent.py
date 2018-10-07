import tensorflow as tf
from config import Config


class Agent:
    def __init__(self):
        print("begin of __init__ of Agent")
        self.X = tf.placeholder(tf.int32, (None,Config.num_fields), "X")
        print("here1")
        self.hide = tf.layers.dense(
            inputs = tf.to_float(self.X, "ToFloat"),
            units = Config.num_fields * 5,
            activation = tf.nn.relu,
            name = "hide")
        print("here2")
        self.logits = tf.layers.dense(
            inputs = self.hide,
            units = Config.num_fields,
            name = "logits")
        self.value = tf.layers.dense(
            inputs = self.hide,
            units = 1,
            name = "value")
        print("here3")
        self.action_prob = tf.nn.softmax(logits=self.logits)
        print("here4")
        self.filter_logits = tf.where(tf.equal(self.X, 0), self.logits, self.logits - 1e9)
        print("here5")
        self.action = tf.multinomial(self.filter_logits, 1)
        print("end of __init__ of Agent")



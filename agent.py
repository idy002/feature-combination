import tensorflow as tf

class Agent:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, (None,), "X")

    def forward(self):
        pass # (action, value, action_prob)


import tensorflow as tf


class Reinforce:
    def __init__(self):
        self.lr = 0.0001
        self.agent = None
        self.Y = tf.placeholder(tf.float32, (None,), "action")
        self.R = tf.placeholder(tf.float32, (None,), "reward")
        self.DR = tf.placeholder(tf.float32, (None,), "discounted_reward")
        self.N = tf.placeholder(tf.float32, (None,), "num_episode")

    def next_batch(self, batch_size, render):
        pass

    def train(self):
        pass

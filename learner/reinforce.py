import tensorflow as tf
import sys
import numpy as np
from actor.actor import Actor
from config import Config

class Reinforce:
    def __init__(self):
        # model
        self.graph = None
        self.sess = None
        self.actor = None

    def build_graph(self):
        sys.stderr.write('Reinforce.build_graph\n')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.actor = Actor()
            self.sess = tf.Session(graph=self.graph)
            writer = tf.summary.FileWriter(Config.reinforce_graph_logdir, graph=self.graph)
            writer.close()


if __name__ == "__main__":
    sys.stderr.write('__module__.start\n')
    learner = Reinforce()
    learner.build_graph()
    sys.stderr.write('__module__.end\n')



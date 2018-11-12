import tensorflow as tf
import numpy as np
import sys
from collections import namedtuple
from learner.actor import Actor
from environment.environment import Enviroment
from learner.actor import State
from config import Config

EpisodeData = namedtuple('EpisodeData', ['fix_combs', 'cur_combs', 'actions', 'rewards', 'discounted_rewards'])
ClassifiedData = namedtuple('ClassifiedData', ['fix_combs', 'cur_combs', 'actions', 'rewards', 'discounted_rewards'])

class Reinforce:
    def __init__(self, learning_rate):
        # model
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter(Config.reinforce_logdir, graph=self.graph)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        with self.graph.as_default():
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32, initializer=tf.zeros_initializer, trainable=False)
            self.actor = Actor(self.optimizer, self.global_step)

    def sample_action(self, state, probs):
        pass

    def trans_data(self, bdata):
        return ClassifiedData()

    def train(self, env, num_batches, batch_size, discount_factor=1.0):
        env = Enviroment()
        for i_batch in range(num_batches):
            bdata = []  # batch data
            for i_episode in range(batch_size):
                state = env.reset()
                edata = EpisodeData([], [], [], [], [])  # episode data
                while True:
                    probs = self.actor.predict(state, self.sess)
                    action = self.sample_action(state, probs)

                    done, next_state, reward = env.step(state, action)

                    edata.fix_combs.append(state.fix_combinations)
                    edata.cur_combs.append(state.cur_combination)
                    edata.actions.append(action)
                    edata.rewards.append(reward)
                    if done:
                        break
                    state = next_state
                len_episode = len(edata.rewards)
                edata.discounted_rewards = list(edata.rewards)
                for i in range(len_episode-2, -1, -1):
                    edata.discounted_rewards[i] += edata.discounted_rewards[i+1] * discount_factor
                bdata.append(edata)
            cdatas = [self.trans_data(bdata)]
            # TODO:
            #   1. finish above two functions,
            #   2. stack the data in cdatas
            #   3. train using cdatas
            #   4. record and output statistics information


if __name__ == "__main__":
    sys.stderr.write('__module__.start\n')
#    learner = Reinforce()
    sys.stderr.write('__module__.end\n')



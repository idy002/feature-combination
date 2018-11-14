import tensorflow as tf
import numpy as np
import sys
from collections import namedtuple
from learner.actor import Actor
from environment.environment import Enviroment
from learner.actor import State
from config import Config

EpisodeData = namedtuple('EpisodeData', ['fix_combs', 'cur_combs', 'actions', 'rewards', 'discounted_rewards'])
ClassifiedData = namedtuple('ClassifiedData', ['fix_combs', 'cur_combs', 'actions', 'discounted_rewards'])


class Reinforce:
    def __init__(self, learning_rate):
        # model
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter(Config.reinforce_logdir, graph=self.graph)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        with self.graph.as_default():
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32, initializer=tf.zeros_initializer, trainable=False)
            self.actor = Actor(self.graph, self.sess, self.optimizer)

    @staticmethod
    def sample_action(state, probs):
        '''
        sample an action according to a probability distribution of actions, the occurred field will be ignored
        :param state: State
        :param probs: (num_fields) ndarray, the probability distribution of actions
        :return: the sampled action from the actions that are not in state.cur_combination
        '''
        num_fail = 0
        while True:
            action = np.random.choice(Config.num_fields, p=probs)
            if state.cur_combination[action] == 0:
                return action
            else:
                num_fail += 1


    @staticmethod
    def trans_data(episode_data_list):
        '''
        transform a list of EpisodeData to a list of ClassifiedData
        every classified data has different lengths of fix_combinations
        :param episode_data_list:
        :return:
        '''
        max_length = 0
        for episodeData in episode_data_list:
            episode_len = len(episodeData.actions)
            for t in range(episode_len):
                length = episodeData.fix_combs[t].shape[0]
                max_length = max(max_length, length)
        cdata = [ClassifiedData([], [], [], []) for l in range(max_length + 1)]
        for episodeData in episode_data_list:
            episode_len = len(episodeData.actions)
            for t in range(episode_len):
                length = episodeData.fix_combs[t].shape[0]
                cdata[length].fix_combs.append(episodeData.fix_combs[t])
                cdata[length].cur_combs.append(episodeData.cur_combs[t])
                cdata[length].actions.append(episodeData.actions[t])
                cdata[length].discounted_rewards.append(episodeData.discounted_rewards[t])
        result = []
        for classifiedData in cdata:
            if len(classifiedData.fix_combs) == 0:
                continue
            result.append(ClassifiedData(
                fix_combs=np.stack(classifiedData.fix_combs),
                cur_combs=np.stack(classifiedData.cur_combs),
                actions=np.stack(classifiedData.actions),
                discounted_rewards=np.stack(classifiedData.discounted_rewards)
            ))
        return result

    def train(self, env, num_batches, batch_size, discount_factor=1.0):
        global_episode_index = 0
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            for i_batch in range(num_batches):
                #   predict
                episode_data_list = []
                for i_episode in range(batch_size):
                    state = env.reset()
                    episode_data = EpisodeData([], [], [], [], [])  # episode data
                    while True:
                        probs, logits = self.actor.predict(state)
                        action = self.sample_action(state, probs[0])

                        done, next_state, reward = env.step(state, action)

                        episode_data.fix_combs.append(state.fix_combinations)
                        episode_data.cur_combs.append(state.cur_combination)
                        episode_data.actions.append(action)
                        episode_data.rewards.append(reward)
                        state = next_state
                        if done:
                            break
                    len_episode = len(episode_data.rewards)
                    episode_data.discounted_rewards.extend(episode_data.rewards)
                    for i in range(len_episode - 2, -1, -1):
                        episode_data.discounted_rewards[i] += episode_data.discounted_rewards[i + 1] * discount_factor
                    episode_data_list.append(episode_data)
                    #  do summary
                    self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=reward)]),
                                            global_step=global_episode_index)
                    print("Episode {}, Reward {:.3f}".format(global_episode_index, reward))
                    print("Selected Field Combinations:\n{}".format(state.fix_combinations))
                    global_episode_index += 1
                #   update
                classified_data_list = self.trans_data(episode_data_list)
                losses = []
                for classified_data in classified_data_list:
                    loss = self.actor.update(classified_data.fix_combs, classified_data.cur_combs,
                                             classified_data.discounted_rewards, classified_data.actions)
                    losses.append(loss)
                mean_loss = np.mean(losses)
                self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=mean_loss)]),
                                        global_step=global_episode_index)
                print("Batch {}, Loss {:.3f}".format(i_batch, mean_loss))
        self.writer.flush()


if __name__ == "__main__":
    sys.stderr.write('__module__.start\n')
    #    learner = Reinforce()
    sys.stderr.write('__module__.end\n')


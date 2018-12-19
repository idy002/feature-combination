import tensorflow as tf
import numpy as np
import sys
import time
from collections import namedtuple
from learner.actor import Actor
from environment.environment import Enviroment
from learner.actor import State
from config import Config

EpisodeData = namedtuple('EpisodeData', ['fix_combs', 'cur_combs', 'actions', 'rewards', 'discounted_rewards', 'auc'])
ClassifiedData = namedtuple('ClassifiedData', ['fix_combs', 'cur_combs', 'actions', 'discounted_rewards'])


class Reinforce:
    def __init__(self, learning_rate):
        # model
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=Config.sess_config)
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
        num_actions = Config.num_fields - np.sum(state.cur_combination)
        return np.random.choice(num_actions, p=probs)


    @staticmethod
    def trans_data(episode_data_list):
        '''
        transform a list of EpisodeData to a list of ClassifiedData
        every (fix_combinations, cur_combination) has the same shape in the same in the same classified data, which
        means the function classified a transition by the shape of fix_combinations and cur_combination
        :param episode_data_list:
        :return: classified data list
        '''
        max_length = 0
        for episodeData in episode_data_list:
            episode_len = len(episodeData.actions)
            for t in range(episode_len):
                length = episodeData.fix_combs[t].shape[0]
                max_length = max(max_length, length)
        cdata = [[ClassifiedData([], [], [], []) for num_fields_cur in range(Config.num_fields+1)]
                    for num_combs_fix in range(max_length + 1)]
        for episodeData in episode_data_list:
            episode_len = len(episodeData.actions)
            for t in range(episode_len):
                num_combs_fix = episodeData.fix_combs[t].shape[0]
                num_fields_cur = int(np.sum(episodeData.cur_combs[t]))

                assert episodeData.actions[t] < Config.num_fields - np.sum(episodeData.cur_combs[t]), 'NO'
                cdata[num_combs_fix][num_fields_cur].fix_combs.append(episodeData.fix_combs[t])
                cdata[num_combs_fix][num_fields_cur].cur_combs.append(episodeData.cur_combs[t])
                cdata[num_combs_fix][num_fields_cur].actions.append(episodeData.actions[t])
                cdata[num_combs_fix][num_fields_cur].discounted_rewards.append(episodeData.discounted_rewards[t])
        result = []
        for classifiedDataList in cdata:
            for classifiedData in classifiedDataList:
                if len(classifiedData.fix_combs) == 0:
                    continue
                result.append(ClassifiedData(
                    fix_combs=np.stack(classifiedData.fix_combs),
                    cur_combs=np.stack(classifiedData.cur_combs),
                    actions=np.stack(classifiedData.actions),
                    discounted_rewards=np.stack(classifiedData.discounted_rewards)
                ))
        return result

    @staticmethod
    def get_elapsed_time(start_time):
        t = time.gmtime(time.time() - start_time)
        return "{hours:02}:{minutes:02}:{seconds:02}".format(hours=(t.tm_yday-1) * 24 + t.tm_hour, minutes=t.tm_min, seconds=t.tm_sec)

    def train(self, env, num_batches, batch_size, discount_factor=1.0):
        global_episode_index = 0
        start_time = time.time()
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            for i_batch in range(num_batches):
                #   predict
                episode_data_list = []
                for i_episode in range(batch_size):
                    state = env.reset()
                    episode_data = EpisodeData([], [], [], [], [], [])  # episode data
                    auc_list = []
                    while True:
                        probs, logits = self.actor.predict(state)
                        action = self.sample_action(state, probs[0])

                        done, next_state, reward, auc = env.step(state, action)

                        if np.sum(next_state.cur_combination) == 0:
                            print("\tPhase: {}  Elapsed: {}  Reward: {:.3f}  Auc: {:.3f}".format(next_state.fix_combinations.shape[0],
                                                                                             self.get_elapsed_time(start_time), reward, auc))

                        episode_data.fix_combs.append(state.fix_combinations)
                        episode_data.cur_combs.append(state.cur_combination)
                        episode_data.actions.append(action)
                        episode_data.rewards.append(reward)
                        episode_data.auc.append(auc)

                        state = next_state
                        if done:
                            break
                    len_episode = len(episode_data.rewards)
                    episode_data.discounted_rewards.extend(episode_data.rewards)
                    for i in range(len_episode - 2, -1, -1):
                        episode_data.discounted_rewards[i] += episode_data.discounted_rewards[i + 1] * discount_factor
                    episode_data_list.append(episode_data)
                    #  do summary
#                    print("Auc: {}".format(episode_data.auc))
                    print("Rewards: {}".format(episode_data.rewards))
                    print("Discounted Rewards: {}".format(episode_data.discounted_rewards))
#                    print("Reward Sum:{},  First Discounted Reward{}".format(np.sum(episode_data.rewards), episode_data.discounted_rewards[0]))
                    print("Batch: {}  Episode {}  Elapsed: {}  Accumulated Reward {:.3f}".format(i_batch, global_episode_index, time.time() - start_time, episode_data.discounted_rewards[0]))
                    print("Selected Field Combinations:\n{}".format(state.fix_combinations[1:]))
                    global_episode_index += 1
                #   update
                classified_data_list = self.trans_data(episode_data_list)
                losses = []
                for classified_data in classified_data_list:
                    loss = self.actor.update(classified_data.fix_combs, classified_data.cur_combs,
                                             classified_data.discounted_rewards, classified_data.actions)
                    losses.append(loss)
                mean_loss = np.mean(losses)
                self.writer.add_summary(tf.Summary(value=[ tf.Summary.Value(tag='Score',
                                        simple_value=np.mean([episode_data.discounted_rewards[0] for episode_data in episode_data_list]))]),
                                        global_step=global_episode_index)
                self.writer.add_summary(tf.Summary(value=[ tf.Summary.Value(tag='Auc',
                                        simple_value=np.mean([episode_data.auc[-1] for episode_data in episode_data_list]))]),
                                        global_step=global_episode_index)
                self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=mean_loss)]),
                                        global_step=global_episode_index)
                print("Batch {}, Loss {:.3f}".format(i_batch, mean_loss))
        self.writer.flush()


if __name__ == "__main__":
    sys.stderr.write('__module__.start\n')
    #    learner = Reinforce()
    sys.stderr.write('__module__.end\n')


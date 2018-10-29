from config import Config
import sys

sys.path.append(Config.data_path)
from datasets import as_dataset
import numpy as np
import itertools # for combinations
from functools import reduce
from sklearn.linear_model import LogisticRegression


class Environment:
    def __init__(self):
        self.dataset = as_dataset(Config.data_name, False)
        self.field_sizes = self.dataset.feat_sizes
        self.iter_fields = range(self.dataset.num_fields)

    def batch_generator(self, **kwargs):
        dataset_batch_generator = self.dataset.batch_generator(kwargs)
        new_feat_sizes = [self.field_sizes[idx] for idx in self.iter_fields]
        num_new_features = reduce(lambda x, y: x * y, new_feat_sizes)
        for X, y in dataset_batch_generator:
            nX = np.zeros((X.shape[0], num_new_features))
            for i in range(X.shape[0]):
                feature_indices = X[i, :]
                sum = 0
                assert feature_indices.size == self.dataset.num_fields, \
                    "Every field must have exactly one feature except %d got %d" % (
                        self.dataset.num_fields, feature_indices.size)
                for j in self.iter_fields:
                    sum = sum * self.field_sizes[j] + feature_indices[j] - self.dataset.feat_min[j]
                nX[i, sum] = 1
            yield nX, y

    def get_state_reward(self, state):
        reward_type = ["hit", "acc"][1]
        if reward_type == "hit":
            return self.check_hit(state)
        else :
            if np.sum(state) < Config.target_num_fields :
                return 0.0

            self.iter_fields = np.concatenate(np.argwhere(state))
            model = LogisticRegression(solver='lbfgs')
            X_all, y_all = [], []
            for X, y in self.batch_generator(gen_type='train', batch_size=10000, on_disk=False):
                X_all.append(X)
                y_all.append(y)
            X_all, y_all = np.concatenate(X_all), np.concatenate(y_all)
            model.fit(X_all, y_all)

            X_all, y_all = [], []
            for X, y in self.batch_generator(gen_type='test', batch_size=10000, on_disk=False):
                X_all.append(X)
                y_all.append(y)
            X_all, y_all = np.concatenate(X_all), np.concatenate(y_all)
            reward = model.score(X_all, y_all)
            # print("fields: {} reward: {}".format(state, reward))
            return reward

    #
    #   compute the reward of a filed combination
    #     - state: one hot state that express the current fields combination
    #     - action: the new field to be added
    #   return:
    #     - (state2, reward): state2 is the new field and reward is the score that the new state get
    #
    def get_reward(self, state, action):
        state2 = np.array(state, np.int32)
        state2[action] = 1
        return state2, self.get_state_reward(state2)

    def check_hit(self, state):
        if isinstance(state,list):
            return 1.0 if state in self.dataset.all_fc else 0.0
        else :
            return 1.0 if np.where(state)[0].tolist() in self.dataset.all_fc else 0.0

if __name__ == "__main__":
    env = Environment()
    print(env.dataset.all_fc)
    for c in itertools.combinations(range(env.dataset.num_fields), Config.target_num_fields):
        state = np.zeros(env.dataset.num_fields, dtype=np.int)
        state[np.array(c)] = 1
        print("{:1.0} {} {}".format(env.check_hit(state), state, env.get_state_reward(state)))


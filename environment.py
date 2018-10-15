from config import Config
import sys

sys.path.append(Config.data_path)
from datasets import as_dataset
import numpy as np
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

    def evaluate(self, fields):
        pass

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
        self.iter_fields = np.concatenate(np.argwhere(state2))
        model = LogisticRegression()
        X_all, y_all = [], []
        for X, y in self.batch_generator(gen_type='train', batch_size=10, on_disk=False):
            X_all.append(X)
            y_all.append(y)
        X_all, y_all = np.concatenate(X_all), np.concatenate(y_all)
        model.fit(X_all, y_all)

        X_all, y_all = [], []
        for X, y in self.batch_generator(gen_type='test', batch_size=10, on_disk=False):
            X_all.append(X)
            y_all.append(y)
        X_all, y_all = np.concatenate(X_all), np.concatenate(y_all)
        reward = model.score(X_all, y_all)
        #print("fields: {} reward: {}".format(state2, reward))
        return state2, reward


if __name__ == "__main__":
    env = Environment()
    for i in range(Config.num_fields):
        for j in range(Config.num_fields):
            if i == j:
                continue
            state = [0, 0, 0, 0]
            state[i] = 1
            action = j
            print(env.get_reward(state, action))

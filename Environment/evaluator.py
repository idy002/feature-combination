from config import Config
from Environment.models import as_model
import numpy as np
import tensorflow as tf


def get_optimizer(name, lr):
    name = name.lower()
    if name == 'sgd' or name == 'gd':
        return tf.train.GradientDescentOptimizer(lr)
    elif name == 'adam':
        return tf.train.AdamOptimizer(lr)
    elif name == 'adamgrad':
        return tf.train.AdagradOptimizer(lr)
    else:
        raise ValueError


class Evaluator:
    def __init__(self):
        self.data_gen_kwargs = dict()
        self.data_gen_kwargs["random_sample"] = True
        self.data_gen_kwargs["val_ratio"] = 0.25
        self.raw_feat_sizes = Config.dataset.feat_sizes
        self.raw_feat_min = Config.dataset.feat_min

        # datainfo
        self.state = None
        self.feat_min = None
        self.feat_sizes = None
        self.num_features = None
        self.num_fields = None

        # model
        self.graph = None
        self.sess = None
        self.model = None
        self.global_step = None
        self.learning_rate = None
        self.optimizer = None
        self.gradients = None
        self.train_op = None
        self.saver = None

        # data generator
        self.train_gen = None
        self.valid_gen = None

    def build_graph(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=0)
            self.learning_rate = tf.get_variable('learning_rate', dtype=tf.float32, initializer=Config.evaluator_learning_rate)
            self.optimizer = get_optimizer(Config.evaluator_optimizer_name, lr=self.learning_rate)
            self.model = as_model(Config.evaluator_model_name, input_dim=self.num_features, num_fields=self.num_fields)
            self.gradients = self.optimizer.compute_gradients(self.model.loss)
            self.train_op = self.optimizer.minimize(self.model.loss, global_step=self.global_step)
            self.saver = tf.train.Saver()

    def init(self, state):
        # init the data information
        self.state = state
        self.feat_min = []
        self.feat_sizes = []
        cur_index = 0
        for i in range(state.shape[0]):
            size = 1
            for field_index in np.where(state[i])[0]:
                size *= self.raw_feat_sizes[field_index]
            self.feat_sizes.append(size)
            self.feat_min.append(cur_index)
            cur_index += size
        self.num_features = cur_index
        self.num_fields = state.shape[0]
        # build the model
        self.build_graph()

    def print_datainfo(self):
        print("state:")
        for i in range(state.shape[0]):
            print("\t", state[i])
        print("feat_sizes:\t", self.feat_sizes)
        print("feat_min:\t", self.feat_min)
        print("num_features:\t", self.num_features)

    def transformX(self, X):
        nX = np.zeros((X.shape[0], self.state.shape[0]))
        for instance_index in range(X.shape[0]):
            for feat_index in range(self.state.shape[0]):
                field_indices = np.where(self.state[feat_index])[0]
                subindex = 0
                for field_index in field_indices:
                    subindex = subindex * self.raw_feat_sizes[field_index] \
                               + X[instance_index, field_index] - self.raw_feat_min[field_index]
                nX[instance_index, feat_index] = self.feat_min[feat_index] + subindex
        return nX

    def batch_generator(self, gen_type, batch_size):
        self.data_gen_kwargs["gen_type"] = gen_type
        self.data_gen_kwargs["batch_size"] = batch_size
        if gen_type == 'train':
            raw_batch_generator = Config.dataset.batch_generator(self.data_gen_kwargs)
            for X, y in raw_batch_generator:
                yield self.transformX(X), y
        elif gen_type == 'valid':
            raw_batch_generator = Config.dataset.batch_generator(self.data_gen_kwargs)
            for X, y in raw_batch_generator:
                yield self.transformX(X), y
        else:
            assert False, "Unknown gen_type"

    def train_batch(self, batch_X, batch_y):
        fetches = [self.train_op, self.model.loss, self.model.log_loss, self.model.l2_loss]
        feed_dict = {self.model.inputs:batch_X, self.model.labels:batch_y, self.model.training:True}
        _, loss, log_loss, l2_loss = self.sess.run(fetches, feed_dict)
        return loss, log_loss, l2_loss

    def score(self, state, max_step=10000):
        self.init(state)
        self.train_gen = self.batch_generator(gen_type="train")
        self.valid_gen = self.batch_generator(gen_type="valid")
        step = 0
        self.sess.run(tf.global_variables_initializer())
        while (max_step is None) or step < max_step:
            pass



if __name__ == "__main__":
    evalator = Evaluator()
    state = np.zeros((Config.target_field_combinations, Config.num_fields), dtype=np.int32)
    for i in range(state.shape[0]):
        samples = np.random.choice(range(Config.num_fields), size=Config.target_field_len, replace=False)
        state[i, samples] = 1
    evalator.init(state)
    evalator.print_datainfo()
    for X, y in evalator.batch_generator(gen_type="train"):
        print(X, y)
    # numfields = Config.num_fields
    # list_states = [[i, j, k] for i in range(numfields) for j in range(i + 1, numfields) for k in
    #                range(j + 1, numfields)]
    # onehot_states = [np.zeros(numfields, dtype=np.int8) for i in range(len(list_states))]
    # for i, list_state in enumerate(list_states):
    #     for j in list_state:
    #         onehot_states[i][j] = 1
    # scores = dict()
    # print(Config.meta)
    # all_fc = Config.meta["field_combinations"]
    # evaluator = Evaluator()
    # for i in range(len(onehot_states)):
    #     for j in range(i + 1, len(onehot_states)):
    #         for k in range(j + 1, len(onehot_states)):
    #             num_hits = 0
    #             num_hits += 1 if list_states[i] in all_fc else 0
    #             num_hits += 1 if list_states[j] in all_fc else 0
    #             num_hits += 1 if list_states[k] in all_fc else 0
    #             if num_hits not in scores:
    #                 scores[num_hits] = []
    #             state = np.stack([onehot_states[i], onehot_states[j], onehot_states[k]])
    #             scores[num_hits].append(evaluator.score(state))
    # for num_hits in scores:
    #     print("num_hits = {} \t cases = {:7d} \t mean_score = {:.3f}".format(
    #         num_hits, len(scores[num_hits]), np.mean(scores[num_hits])))

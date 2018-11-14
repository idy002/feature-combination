from config import Config
import sys
from environment.models import as_model
from common import get_optimizer
import time
import sklearn
import random
import numpy as np
import tensorflow as tf
import itertools
import math


class StateDatasetIterator:
    def __init__(self, evaluator, kwargs):
        self.evaluator = evaluator
        self.kwargs = kwargs

    def __iter__(self):
        for batch_xs, batch_ys in Config.dataset.batch_generator(kwargs=self.kwargs):
            yield self.evaluator.transformX(batch_xs), batch_ys


class Evaluator:
    def __init__(self):
        # general
        self.render = None

        # raw dataset information
        self.raw_feat_sizes = Config.dataset.feat_sizes
        self.raw_feat_min = Config.dataset.feat_min

        # dataset information
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

        # train and evaluate
        self.start_time = None
        self.train_writer = None
        self.valid_writer = None
        self.test_writer = None

    def build_graph(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(config=Config.sess_config, graph=self.graph)
        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=0)
            self.learning_rate = tf.get_variable('learning_rate', dtype=tf.float32,
                                                 initializer=Config.evaluator_learning_rate)
            self.optimizer = get_optimizer(Config.evaluator_optimizer_name, lr=self.learning_rate)
            self.model = as_model(Config.evaluator_model_name, input_dim=self.num_features, num_fields=self.num_fields)
            self.gradients = self.optimizer.compute_gradients(self.model.loss)
            self.train_op = self.optimizer.minimize(self.model.loss, global_step=self.global_step)
            self.saver = tf.train.Saver()
            if self.render:
                self.train_writer = tf.summary.FileWriter(Config.evaluator_train_logdir, graph=self.graph, flush_secs=10.0)
                self.valid_writer = tf.summary.FileWriter(Config.evaluator_valid_logdir, graph=self.graph, flush_secs=10.0)
                graph_writer = tf.summary.FileWriter(Config.evaluator_graph_logdir, graph=self.graph, flush_secs=10.0)
                graph_writer.close()

    def init_dataset(self, state):
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
        self.train_gen = self.batch_generator(gen_type="train", batch_size=100)
        self.valid_gen = self.batch_generator(gen_type="valid", batch_size=100)

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
        data_gen_kwargs = dict()
        data_gen_kwargs["random_sample"] = True
        data_gen_kwargs["val_ratio"] = 0.25
        data_gen_kwargs["gen_type"] = gen_type
        data_gen_kwargs["squeeze_output"] = False
        data_gen_kwargs["batch_size"] = batch_size
        data_gen_kwargs["on_disk"] = False
        return StateDatasetIterator(self, data_gen_kwargs)

    def train_batch(self, batch_xs, batch_ys):
        fetches = [self.train_op, self.model.loss, self.model.log_loss, self.model.l2_loss]
        feed_dict = {self.model.inputs: batch_xs, self.model.labels: batch_ys, self.model.training: True}
        _, loss, log_loss, l2_loss = self.sess.run(fetches, feed_dict)
        return loss, log_loss, l2_loss

    def get_elapsed(self):
        return time.time() - self.start_time

    def train(self, state, max_rounds=100, log_step_frequency=10, eval_round_frequency=1, early_stop_rounds=3,
              render=True):
        """
        train the model given the specified state, use auc as the performance.
        """
        with self.graph.as_default():
            self.train_gen = self.batch_generator(gen_type="train", batch_size=1000)
            step = 0
            round = 0
            self.sess.run(tf.global_variables_initializer())
            self.start_time = time.time()
            all_auc = []
            while round < max_rounds:
                if render:
                    print("Round: {}".format(step))
                for batch_xs, batch_ys in self.train_gen:
                    loss, log_loss, l2_loss = self.train_batch(batch_xs, batch_ys)
                    step = self.sess.run(self.global_step)
                    if render and step % log_step_frequency == 0:
                        print("Done step {:2d}, Elapsed: {:.3f} seconds, Loss: {:.3f}, Log-Loss: {:.3f}, L2-Loss: {:.3f}"
                              .format(step, self.get_elapsed(), loss, log_loss, l2_loss))
                        summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss),
                                                    tf.Summary.Value(tag='log_loss', simple_value=log_loss),
                                                    tf.Summary.Value(tag='l2_loss', simple_value=l2_loss)])
                        self.train_writer.add_summary(summary, global_step=step)
                if round % eval_round_frequency == 0:
                    log_loss, auc = self.evaluate(self.valid_gen, self.valid_writer)
                    if render:
                        print("Round {:2d}, Elapsed: {:.3f} seconds, Log-Loss: {:.3f}, Auc: {:.3f}"
                              .format(round, self.get_elapsed(), log_loss, auc))
                    all_auc.append(auc)
                    max_auc = max(all_auc)
                    if max(all_auc[-early_stop_rounds:]) < max_auc:
                        return
                round += 1

    def evaluate_batch(self, batch_xs, batch_ys):
        fetches = self.model.preds
        feed_dict = {self.model.inputs: batch_xs, self.model.labels: batch_ys, self.model.training: False}
        return self.sess.run(fetches=fetches, feed_dict=feed_dict)

    '''
    return log_loss, auc
    '''

    def evaluate(self, gen, writer):
        labels = []
        preds = []
        for batch_xs, batch_ys in gen:
            labels.append(batch_ys)
            preds.append(self.evaluate_batch(batch_xs, batch_ys))
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        preds = np.clip(preds, 1e-6, 1 - 1e-6)
        log_loss = sklearn.metrics.log_loss(y_true=labels, y_pred=preds)
        auc = sklearn.metrics.roc_auc_score(y_true=labels, y_score=preds)
        return log_loss, auc

    def score(self, state, render=False):
        self.render = render
        self.init_dataset(state)
        self.build_graph()
        self.train(state, max_rounds=300, log_step_frequency=10, eval_round_frequency=1, early_stop_rounds=3,
                   render=render)
        log_loss, auc = self.evaluate(self.valid_gen, self.valid_writer)
        return auc


if __name__ == "__main__":
    print(Config.meta)
    key_fc = Config.meta["field_combinations"]
    num_key_fc = len(key_fc)
    len_fc = len(key_fc[0])
    num_fields = Config.num_fields

    print("key_fc: {}".format(key_fc))
    print("num_key_fc: {}".format(num_key_fc))
    print("num_fields: {}".format(num_fields))
    print("len_fc: {}".format(len_fc))
    list_fc = [a for a in itertools.combinations(range(num_fields), len_fc)]
    num_fc = 0
    for i in list_fc:
        num_fc += 1
    print("num_fc: {}".format(num_fc))
    onehot_fc_notkey = []
    onehot_fc_key = []
    for i in range(len(list_fc)):
        onehot_fc = np.zeros(num_fields)
        for j in list_fc[i]:
            onehot_fc[j] = 1
        if list(list_fc[i]) in key_fc:
            onehot_fc_key.append(onehot_fc)
        else:
            onehot_fc_notkey.append(onehot_fc)

    evaluator = Evaluator()
    scores = [[] for i in range(num_key_fc + 1)]
    covers = [[] for i in range(num_key_fc + 1)]
    fmean = lambda x: np.mean(x) if len(x) > 0 else np.NaN
    fstddev = lambda x: np.std(x, ddof=1) if len(x) > 1 else np.NaN
    while True:
        a = random.randint(0, num_key_fc)
        b = num_key_fc - a
        cur_fc_list = []
        cur_fc_list.extend(random.sample(onehot_fc_key, a))
        cur_fc_list.extend(random.sample(onehot_fc_notkey, b))
        state = np.stack(cur_fc_list, axis=0)
        scores[a].append(evaluator.score(state, render=True))
        covers[a].append(len(np.where(sum(cur_fc_list) > 0)[0]))
        print('\r', end="")
        for i in range(num_key_fc + 1):
            print("{:.4f}({:.2f},{:.2f})  ".format(fmean(scores[i]), fstddev(scores[i]), fmean(covers[i])), end="")
            sys.stdout.flush()

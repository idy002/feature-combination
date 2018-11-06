from config import Config
from Environment.models import as_model
import time
import sklearn
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

class StateDatasetIterator:
    def __init__(self, evaluator, kwargs):
        self.evaluator = evaluator
        self.kwargs = kwargs

    def __iter__(self):
        for batch_xs, batch_ys in Config.dataset.batch_generator(kwargs=self.kwargs):
            yield evaluator.transformX(batch_xs), batch_ys

class Evaluator:
    def __init__(self):
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
            self.train_writer = tf.summary.FileWriter(Config.evaluator_train_logdir, graph=self.graph, flush_secs=10.0)
            self.valid_writer = tf.summary.FileWriter(Config.evaluator_valid_logdir, graph=self.graph, flush_secs=10.0)

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
        return StateDatasetIterator(self, data_gen_kwargs)

    def train_batch(self, batch_xs, batch_ys):
        fetches = [self.train_op, self.model.loss, self.model.log_loss, self.model.l2_loss]
        feed_dict = {self.model.inputs: batch_xs, self.model.labels: batch_ys, self.model.training: True}
        _, loss, log_loss, l2_loss = self.sess.run(fetches, feed_dict)
        return loss, log_loss, l2_loss

    def get_elapsed(self):
        return time.time() - self.start_time

    '''
    train the model given the specified state, use auc as the performance. 
    '''

    def train(self, state, max_rounds=100, log_step_frequency=10, eval_round_frequency=1, early_stop_rounds=3,
              render=True):
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
                        print("earlier stop!")
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
        log_loss = sklearn.metrics.log_loss(y_true=labels, y_pred=preds)
        auc = sklearn.metrics.roc_auc_score(y_true=labels, y_score=preds)
        return log_loss, auc

    def score(self, state, render=False):
        self.init_dataset(state)
        self.build_graph()
        self.train(state, max_rounds=100, log_step_frequency=10, eval_round_frequency=1, early_stop_rounds=3,
                   render=render)
        log_loss, auc = self.evaluate(self.valid_gen, self.valid_writer)
        return auc


if __name__ == "__main__":
    #    evaluator = Evaluator()
    #    state = np.zeros((Config.target_combination_num, Config.num_fields), dtype=np.int32)
    #    for i in range(state.shape[0]):
    #        samples = np.random.choice(range(Config.num_fields), size=Config.target_combination_len, replace=False)
    #        state[i, samples] = 1
    #    evaluator.init_dataset(state)
    #    evaluator.print_datainfo()
    #    for X, y in evaluator.batch_generator(gen_type="train"):
    #        print(X, y)
    numfields = Config.num_fields
    list_states = [[i, j, k] for i in range(numfields) for j in range(i + 1, numfields) for k in
                   range(j + 1, numfields)]
    onehot_states = [np.zeros(numfields, dtype=np.int8) for i in range(len(list_states))]
    for i, list_state in enumerate(list_states):
        for j in list_state:
            onehot_states[i][j] = 1
    scores = dict()
    print(Config.meta)
    all_fc = Config.meta["field_combinations"]
    evaluator = Evaluator()
    for i in range(len(onehot_states)):
        for j in range(i + 1, len(onehot_states)):
            for k in range(j + 1, len(onehot_states)):
                num_hits = 0
                num_hits += 1 if list_states[i] in all_fc else 0
                num_hits += 1 if list_states[j] in all_fc else 0
                num_hits += 1 if list_states[k] in all_fc else 0
                if num_hits not in scores:
                    scores[num_hits] = []
                state = np.stack([onehot_states[i], onehot_states[j], onehot_states[k]])
                score = evaluator.score(state, render=False)
                scores[num_hits].append(score)
                print("State: {}  Num_hit: {} Score: {:3f}".format([i, j, k], num_hits, score))
    for num_hits in scores:
        print("num_hits = {} \t cases = {:7d} \t mean_score = {:.3f}".format(
            num_hits, len(scores[num_hits]), np.mean(scores[num_hits])))

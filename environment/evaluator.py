import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np
import time
from config import Config
from environment.models import as_model
import matplotlib.pyplot as plt
import sys


class Evaluator:
    def _get_data(self, use_ratio=0.1, train_ratio=0.8):
        X_train = Config.dataset.X_train
        y_train = Config.dataset.y_train
        y_train = np.reshape(y_train, [-1])
        tot_samples = X_train.shape[0]
        num_small_train = int(tot_samples * use_ratio * train_ratio)
        num_small_valid = int(tot_samples * use_ratio * (1.0 - train_ratio))
        X_small_train = X_train[:num_small_train]
        y_small_train = y_train[:num_small_train]
        X_small_valid = X_train[num_small_train:num_small_train + num_small_valid]
        y_small_valid = y_train[num_small_train:num_small_train + num_small_valid]
        return X_small_train, y_small_train, X_small_valid, y_small_valid

    @staticmethod
    def print(round, cur_sample, tot_samples, start_time, train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc):
        progress = "".join([('#' if i *  tot_samples < cur_sample * 30 else '=') for i in range(30)])
        status = "\r\tRound:{} {}/{} [{}] Elapsed: {:.3f} seconds, train_loss:{:.3f} train_acc:{:.3f} train_auc:{:.3f} valid_loss:{:.3f} valid_acc:{:.3f} valid_auc:{:.3f}".format(
            round, cur_sample, tot_samples, progress, time.time() - start_time, train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc
        )
        sys.stdout.write(status)
        if cur_sample == tot_samples:
            sys.stdout.write("\n")

    def evaluate(self, model_name, max_rounds=50, use_ratio=0.4, lr=0.001, **model_params):
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        with graph.as_default():
            learning_rate = tf.get_variable('lr', dtype=tf.float32, initializer=lr, trainable=False)
            model = as_model(model_name, **model_params)
            return
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = opt.minimize(model.loss)

        num_rounds = max_rounds
        batch_size = 100
        eval_per_steps = 30

        X_train, y_train, X_valid, y_valid = self._get_data(use_ratio=use_ratio, train_ratio=0.8)
        tot_samples = X_train.shape[0]
        tot_steps = (tot_samples + batch_size - 1) // batch_size
        eval_per_steps = min(eval_per_steps, tot_steps)

        history = {'train_loss': [], 'train_acc': [], 'train_auc': [], 'valid_loss': [], 'valid_acc': [], 'valid_auc': []}

        with graph.as_default():
            sess.run(tf.global_variables_initializer())
            for round in range(num_rounds):
                start_time = time.time()
                train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for step in range(tot_steps):
                    inputs = X_train[step * batch_size: (step + 1) * batch_size]
                    labels = y_train[step * batch_size: (step + 1) * batch_size]
                    _, logits, loss = sess.run((train_op, model.logits, model.loss), feed_dict={model.inputs: inputs, model.labels: labels})
                    preds = np.where(logits > 0, np.ones_like(logits, dtype=np.int32), np.zeros_like(logits, dtype=np.int32))
                    train_loss = (train_loss * step + np.mean(loss)) / (step + 1)
                    train_auc = (train_auc * step + roc_auc_score(labels, logits)) / (step + 1)
                    train_acc = (train_acc * step + np.count_nonzero(labels == preds) / batch_size) / (step + 1)

                    if (step + 1) % eval_per_steps == 0 or step + 1 == tot_steps or step == 0:
                        val_logits, val_loss = sess.run((model.logits, model.loss), feed_dict={model.inputs: X_valid, model.labels: y_valid})
                        val_labels = y_valid
                        val_preds = np.where(val_logits > 0, np.ones_like(val_logits, dtype=np.int32), np.zeros_like(val_logits, dtype=np.int32))
                        valid_loss = np.mean(val_loss)
                        valid_acc = np.count_nonzero(val_labels == val_preds) / X_valid.shape[0]
                        valid_auc = roc_auc_score(val_labels, val_logits)
                    self.print(round, min((step + 1) * batch_size, tot_samples), tot_samples, start_time, train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc)
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['train_auc'].append(train_auc)
                history['valid_loss'].append(valid_loss)
                history['valid_acc'].append(valid_acc)
                history['valid_auc'].append(valid_auc)
        return history

def plot_histories(histories, info):
    # Plot training & validation accuracy values
    legends = []
    plt.title('Models accuracy' + '(' + info + ')')
    filename = "figures/figure(" + info + ').png'
    for model_name, points in histories.items():
        plt.plot(points)
        legends.append(model_name)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(legends, loc='upper left')
    plt.savefig(filename, format='png')
    plt.show()

if __name__ == '__main__':
    evaluator = Evaluator()
    nn_layers = [(100, 'relu'), (20, 'relu'), (1, None)]
    histories = {}

    use_ratio = 0.3
    lr = 0.0001
    max_rounds = 200

    intersections = Config.dataset.meta['field_combinations']
    h = evaluator.evaluate('fcomb', max_rounds=max_rounds, use_ratio=use_ratio, intersections=intersections, lr=lr, nn_layers=nn_layers)
#    histories['fcomb'] = h['valid_auc']
#    print("fcomb:{:.3f}".format(h['valid_auc'][-1]))

    h = evaluator.evaluate('pin', max_rounds=max_rounds, use_ratio=use_ratio, lr=lr, nn_layers=nn_layers)
#    histories['pin'] = h['valid_auc']
#    print("pin:{:.3f}".format(h['valid_auc'][-1]))

    h = evaluator.evaluate('pnn', max_rounds=max_rounds, use_ratio=use_ratio, lr=lr, nn_layers=nn_layers)
#    histories['pnn'] = h['valid_auc']
#    print("pnn:{:.3f}".format(h['valid_auc'][-1]))

    plot_histories(histories, "dataset:{};rounds:{};use_ratio:{:.3f};lr:{:.5f}".format(Config.data_name, max_rounds, use_ratio, lr))


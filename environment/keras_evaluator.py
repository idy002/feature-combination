import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model
import numpy as np

from config import Config



class Evaluator:
    def evaluate_state(self, raw_state):
        state = []
        for i in range(raw_state.shape[0]):
            state.append(np.where(raw_state[i])[0].tolist())
        print(state)
        loss, acc, history = self.evaluate(
            model_name='fcomb',
            combine_type='seq',
            dense_layers=[(100, 'relu'), (20, 'relu'), (1, 'sigmoid')],
            state=state,
            verbose=1)
        return acc

    def evaluate(self, model_name, verbose=0, use_ratio=1.0, lr=0.001, decay_rate=0.95, **kwargs):
        """
        evaluate a model
        """
        model = self._build_model(model_name, **kwargs)
        model = multi_gpu_model(model=model, gpus=4, cpu_relocation=True)
        model.compile(
            optimizer=keras.optimizers.Adam(lr=lr),
            loss=keras.losses.binary_crossentropy,
            metrics=['binary_accuracy']
        )
        X_small_train, y_small_train, X_small_valid, y_small_valid = self._get_data(use_ratio=use_ratio, train_ratio=0.8)
        result = model.fit(
            x=X_small_train,
            y=y_small_train,
            batch_size=128,
            epochs=100,
            validation_data=[X_small_valid, y_small_valid],
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', min_delta=1e-3, patience=5),
                keras.callbacks.LearningRateScheduler(lambda epoch, old_lr: old_lr * decay_rate, 1)
            ]
        )
        acc = max(result.history['val_binary_accuracy'])
        loss = min(result.history['val_loss'])
        print(model_name, ": ", acc)
        return loss, acc, result.history

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

    def _build_model(self, model_name, **kwargs):
        if model_name == 'lr':
            return LR()
        elif model_name == 'fcomb':
            return FCOMB(**kwargs)
        elif model_name == 'deepfm':
            return DEEPFM(**kwargs)
        elif model_name == 'kpnn':
            kwargs['state'] = [[i, j] for i in range(Config.num_fields) for j in range(i + 1, Config.num_fields)]
            kwargs['combine_type'] = 'seq'
            return FCOMB(**kwargs)
        elif model_name == 'fnn':
            return FNN(**kwargs)
        else:
            raise ValueError


class LR(keras.Model):
    def __init__(self):
        super(LR, self).__init__(name='LR')
        self.embed = layers.Embedding(input_dim=np.sum(Config.feat_sizes),
                                      output_dim=1,
                                      input_length=Config.num_fields)

    def call(self, inputs):  # (None, num_fields)
        embeded = self.embed(inputs)  # (None, num_fields, 1)
        output = tf.sigmoid(tf.reduce_mean(embeded, axis=[1]))
        return output


class FCOMB(keras.Model):
    def __init__(self, state, dense_layers, combine_type):
        super(FCOMB, self).__init__(name='FCOMB')
        self.state = state
        self.embed = layers.Embedding(input_dim=np.sum(Config.feat_sizes),
                                      output_dim=Config.evaluator_embedding_size,
                                      input_length=Config.num_fields)
        self.combine_layers = []
        self.dense_layers = []
        for i in range(len(state)):
            if combine_type == 'seq':
                layer = SeqCombineLayer(state[i], Config.evaluator_embedding_size)
            elif combine_type == 'set':
                layer = SetCombineLayer(state[i])
            else:
                raise ValueError
            self.combine_layers.append(layer)
        for layer_units, activation in dense_layers:
            self.dense_layers.append(layers.Dense(layer_units, activation))

    def call(self, inputs):
        print("FCOMB.call")
        embeded = self.embed(inputs)
        combined = []
        for combine_layer in self.combine_layers:
            combined.append(combine_layer(embeded))
        combined.append(layers.Flatten()(embeded))
        concatenated = tf.concat(combined, axis=1)
        cur_layer = concatenated
        for dense_layer in self.dense_layers:
            cur_layer = dense_layer(cur_layer)
        return cur_layer


class FNN(keras.Model):
    def __init__(self, dense_layers):
        super(FNN, self).__init__(name='FNN')
        self.embed = layers.Embedding(input_dim=np.sum(Config.feat_sizes),
                                      output_dim=Config.evaluator_embedding_size,
                                      input_length=Config.num_fields)
        self.dense_layers = []
        for layer_units, activation in dense_layers:
            self.dense_layers.append(layers.Dense(layer_units, activation))

    def call(self, inputs):
        embeded = self.embed(inputs)  # (None, num_fields, embed_size)
        concatenated = K.reshape(embeded, [-1, int(embeded.shape[1]) * int(embeded.shape[2])])
        cur_layer = concatenated
        for dense_layer in self.dense_layers:
            cur_layer = dense_layer(cur_layer)
        return cur_layer


class DEEPFM(keras.Model):
    def __init__(self, dense_layers):
        super(DEEPFM, self).__init__(name='DEEPFM')
        self.weight_embed = layers.Embedding(input_dim=np.sum(Config.feat_sizes),
                                             output_dim=1,
                                             input_length=Config.num_fields)
        self.vector_embed = layers.Embedding(input_dim=np.sum(Config.feat_sizes),
                                             output_dim=Config.evaluator_embedding_size,
                                             input_length=Config.num_fields, embeddings_initializer=tf.initializers.truncated_normal())
        self.dense_layers = []
        for i in range(len(dense_layers)):
            layers_units = dense_layers[i][0]
            activation = dense_layers[i][1]
            self.dense_layers.append(layers.Dense(layers_units, activation=(activation if i < len(dense_layers) - 1 else None)))

    def call(self, inputs):
        weight_embeded = self.weight_embed(inputs)  # (None, num_fields, 1)
        vector_embeded = self.vector_embed(inputs)  # (None, num_fields, embed_size)
        output_1 = tf.reduce_sum(weight_embeded, axis=[1, 2])
        output_1 = tf.expand_dims(output_1, axis=1)
        output_2 = tf.reduce_sum(tf.square(tf.reduce_sum(vector_embeded, axis=1)), axis=1) \
                   - tf.reduce_sum(tf.square(vector_embeded), axis=[1, 2])
        output_2 = tf.expand_dims(output_2, axis=1)
        cur_layer = tf.reshape(vector_embeded, shape=[-1, Config.num_fields * Config.evaluator_embedding_size])
        for dense_layer in self.dense_layers:
            cur_layer = dense_layer(cur_layer)
        output_3 = cur_layer
        output = tf.sigmoid(output_1 + output_2 + output_3)
        return output


class OuterProductLayer(keras.layers.Layer):
    """
    input_shape: (None, dima), (None, dimb)
    output_shape: (None, dima * dimb) or (None, dima * dimb + dima + dimb) or (None, dense_units)
    """

    def __init__(self, concat_flag, dense_layer_config, **kwargs):
        """
        :param concat_flag: whether add concatenation of inputs
        :param dense_layer:  None for no dense layer or (int, str) for additional dense layer (layer_units, activation)
        :param kwargs:
        """
        self.concat_flag = concat_flag
        self.dense_layer_config = dense_layer_config
        self.dense_layer = None
        super(OuterProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 2
        dense_input_dim = int(input_shape[0][1]) * int(input_shape[1][1])
        if self.concat_flag:
            dense_input_dim += int(input_shape[0][1]) + int(input_shape[1][1])
        if self.dense_layer is not None:
            layer_units, activation = self.dense_layer_config
            self.dense_layer = layers.Dense(units=layer_units, activation=activation)
        super(OuterProductLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, y = inputs
        x = K.expand_dims(x, axis=2)
        y = K.expand_dims(y, axis=1)
        z = layers.Flatten()(x * y)
        if self.concat_flag:
            z = tf.concat([z, inputs[0], inputs[1]], axis=1)
        if self.dense_layer is not None:
            z = self.dense_layer(z)
        return z

#    def compute_output_shape(self, input_shape):
#        return input_shape[0][-1] * input_shape[1][-1]

def outer_product_concat_layer(inputs, dense_units, activation):
    x, y = inputs
    x = K.expand_dims(x, axis=2)
    y = K.expand_dims(y, axis=1)
    z = layers.Flatten()(x * y)
    z = tf.concat([z, inputs[0], inputs[1]], axis=1)
    z = layers.Dense(dense_units, activation)(z)
    return z

def pin_layer(inputs, units_1, units_2):
    x, y = inputs
    z = tf.concat([x, y, x * y], axis=1)
    return layers.Dense(units_2, None)(layers.Dense(units_1, 'relu')(z))

def concat_dense_layer(inputs):
    x, y = inputs
    z = tf.concat([x, y], axis=1)
    return layers.Dense(Config.evaluator_embedding_size, 'relu')(z)

class SetCombineLayer(keras.layers.Layer):
    """
    input_shape: (None, num_fields, embed_size)
    output_shape: (None, output_dim) where output_dim = multiply_reduce(field sizes)
    """

    def __init__(self, fields, **kwargs):
        self.fields = fields
        self.kernel = None
        super(SetCombineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=[self.compute_output_shape(input_shape)])
        super(SetCombineLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        cur = x[:, self.fields[0], :]
        for i in range(1, len(self.fields)):
            cur = OuterProductLayer(concat_flag=False, dense_layer_config=None)([cur, x[:, self.fields[i], :]])
        cur = cur * self.kernel
        return cur

    def compute_output_shape(self, input_shape):
        return np.power(int(input_shape[2]), len(self.fields))


class SeqCombineLayer(keras.layers.Layer):
    """
    input_shape: (None, num_fields, embed_size)
    output_shape: (None, output_dim) where output_dim = cdims[-1]
    """

    def __init__(self, fields, cdims, **kwargs):
        self.fields = fields
        if isinstance(cdims, list):
            assert len(cdims) == len(fields) - 1
            self.cdims = cdims
        else:
            self.cdims = [cdims for _ in range(len(fields) - 1)]
        super(SeqCombineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SeqCombineLayer, self).build(input_shape)

    def call(self, x, **kwargs):
#        cur = x[:, self.fields[0], :]
        inputs = []
        for i in range(len(self.fields)):
            inputs.append(x[:, self.fields[i], :])
        next_inputs = []
        outputs = []
        while len(inputs) > 1:
            for i in range(1, len(inputs), 2):
                dim = int(max(int(inputs[i-1].shape[1]), int(inputs[i].shape[1])))
                new_concat = outer_product_concat_layer([inputs[i-1], inputs[i]], Config.evaluator_embedding_size, 'relu')
                next_inputs.append(new_concat)
                outputs.append(new_concat)
                inputs[i-1] = None
                inputs[i] = None
            if inputs[-1] != None:
                next_inputs.append(inputs[-1])
            inputs = next_inputs
            next_inputs = []
        output = tf.concat(outputs, axis=1)
        return output
#        for i in range(1, len(self.fields)):
#            cur = concat_dense_layer([cur, x[:, self.fields[i], :]])
#            cur = pin_layer([cur, x[:, self.fields[i], :]], Config.evaluator_embedding_size, Config.evaluator_embedding_size)
#            cur = outer_product_concat_layer([cur, x[:, self.fields[i], :]], self.cdims[i-1], 'relu')
#            out = OuterProductLayer(concat_flag=False, dense_layer_config=None)([cur, x[:, self.fields[i], :]])
#            cur = layers.Dense(self.cdims[i-1], 'relu')(out)
#        return cur

    def compute_output_shape(self, input_shape):
        return self.cdims[-1]

def plot_histories(histories):
    # Plot training & validation accuracy values
    legends = []
    filename = "figure"
    for model_name, points in histories.items():
        if model_name == 'note':
            info = points
            plt.title('Models accuracy' + '(' + info + ')')
            filename = "figures/figure(" + info + ').png'
            continue
        plt.plot(points)
        legends.append(model_name)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(legends, loc='upper left')
    plt.savefig(filename, format='png')
#    plt.show()

if __name__ == '__main__':
    evaluator = Evaluator()
    state = []
    T = 5
    for t in range(T):
        perm = list(range(Config.num_fields))
        random.shuffle(perm)
        state.append(perm)
    histories = {}
    use_ratio = 1.0
    lr = 0.0005
    decay_ratio = 0.98
    loss, acc, history = evaluator.evaluate(model_name='fcomb', combine_type='seq', dense_layers=[(200, 'relu'), (40, 'relu'), (1, 'sigmoid')], state=state, verbose=1, use_ratio=use_ratio)
    histories['fcomb'] = history['val_binary_accuracy']
    loss, acc, history = evaluator.evaluate(model_name='kpnn', dense_layers=[(200, 'relu'), (40, 'relu'), (1, 'sigmoid')], verbose=1, use_ratio=use_ratio)
    histories['kpnn'] = history['val_binary_accuracy']
    loss, acc, history = evaluator.evaluate(model_name='fnn', dense_layers=[(200, 'relu'), (40, 'relu'), (1, 'sigmoid')], verbose=1, use_ratio=use_ratio)
    histories['fnn'] = history['val_binary_accuracy']
#    loss, acc, history = evaluator.evaluate(model_name='deepfm', dense_layers=[(100, 'relu'), (20, 'relu'), (1, None)], verbose=1)
#    histories['deepfm'] = history['val_binary_accuracy']
    loss, acc, history = evaluator.evaluate(model_name='lr', verbose=1, use_ratio=use_ratio)
    histories['lr'] = history['val_binary_accuracy']
    histories['note'] = 'opt:adam;lr:{:.3f};use_ratio:{:.3f};decay_ratio:{:.3f}'.format(lr, use_ratio, decay_ratio)
    plot_histories(histories)


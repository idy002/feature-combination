import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import numpy as np
layers.concatenate
import tensorflow as tf
from config import Config


class Evaluator:
    def evaluate(self, state, type):
        """
        evaluate the state
        :param state: [[f1, f2, ..., fk_1], ...], the state: a set of field combinations
        :param type: 'seq' or 'set'
        :return: the score of state, higher score means better state
        """
        model = self._build_model(state, type)
        model.summary()
        X_train = Config.dataset.X_train
        y_train = Config.dataset.y_train
        ratio = 0.8
        tot_train = X_train.shape[0]
        num_train = int(X_train.shape[0] * ratio)
        num_validation = int(X_train.shape[0] * (1.0 - ratio))
        result = model.fit(x=X_train[:num_train], y=y_train[:num_train], batch_size=32, epochs=10,
                  validation_data=[X_train[-num_validation:], y_train[-num_validation:]], verbose=1)
        print(result.history)

    def _build_model(self, state, type):
        if type == 'set':
            input = layers.Input(shape=[Config.num_fields], dtype=tf.int32)  # batch * num_fields
            embeded = layers.Embedding(input_dim=np.sum(Config.feat_sizes),
                                       output_dim=Config.evaluator_embedding_size,
                                       input_length=Config.num_fields)(input)  # batch * num_fields * embed_size
#            combined = []
#            for i in range(len(state)):
#                ith_combined = SetCombineLayer(state[i])([embeded[j] for j in state[i]])
#                combined.append(ith_combined)
#            concatenated = layers.concatenate(combined)
            h1 = layers.Dense(100, 'relu')( layers.Flatten()(embeded) )
            h2 = layers.Dense(20, 'relu')(h1)
            output = layers.Dense(1, 'sigmoid')(h2)
            #fc_layers = [(100, 'relu'), (20, 'relu'), (1, 'sigmoid')]
            #curlayer = concatenated
            #for layer_size, activation in fc_layers:
            #    curlayer = layers.Dense(layer_size, activation=activation)(curlayer)
            #output = curlayer
            model = Model(inputs=input, outputs=output)
            model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.binary_crossentropy, metrics=['binary_accuracy'])
            return model
        else:
            pass


class OuterProductLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OuterProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 2

        super(OuterProductLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, y = inputs
        x = K.expand_dims(x, axis=2)
        y = K.expand_dims(y, axis=1)
        z = x * y
        return K.batch_flatten(z)

    def compute_output_shape(self, input_shape):
        return input_shape[0][-1] * input_shape[1][-1]


class SetCombineLayer(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.kernel = None
        super(SetCombineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.kernel = self.add_weight(name='kernel', shape=[self.compute_output_shape(input_shape)])
        super(SetCombineLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        cur = x[0]
        for i in range(1, len(x)):
            cur = OuterProductLayer()([cur, x[i]])
        cur = cur * self.kernel
        return cur

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return np.multiply.reduce([int(shape[1]) for shape in input_shape])


class SeqCombineLayer(keras.layers.Layer):
    def __init__(self, field_set, **kwargs):
        self.field_set = field_set
        super(SeqCombineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x):
        pass

    def compute_output_shape(self, input_shape):
        pass

if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.evaluate([[i] for i in range(Config.num_fields)], 'set')


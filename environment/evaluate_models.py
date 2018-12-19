import tensorflow as tf
from utility import get_initializer

class Model:
    def __init__(self, input_dim, num_fields):
        self.inputs = None  # batch * num_fields
        self.labels = None  # batch
        self.preds = None  # batch
        self.embeded = None  # batch * num_fields * embed_size
        self.nn_inputs = None  # batch * nn_input_size
        self.logits = None  # batch
        self.loss = None  # batch

        self.input_dim = input_dim
        self.num_fields = num_fields

    def define_placeholder(self):
        with tf.variable_scope("input"):
            self.inputs = tf.placeholder(tf.int32, shape=[None, self.num_fields], name="inputs")  # batch * input_dim
            self.labels = tf.placeholder(tf.float32, shape=[None, self.output_dim], name="labels")  # batch * 1

    def define_embedding(self):
        with tf.variable_scope("embedding"):
            initializer = get_initializer(init_type="xavier")
            w = tf.get_variable("w", shape=[self.input_dim, embed_size], dtype=tf.float32, initializer=initializer,
                                collections=EMBEDS)
            self.embed = tf.nn.embedding_lookup(w, self.inputs)  # batch * fields * embed_size

    def define_interactions(self):
        pass

    def define_nn(self):
        pass

    def define_loss(self):
        pass








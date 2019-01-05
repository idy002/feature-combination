import tensorflow as tf
from config import Config
from utility import get_initializer, get_activation


def as_model(model_name, **model_params):
    model_name = model_name.lower()
    input_dim = sum(Config.feat_sizes)
    num_fields = Config.num_fields
    model = None
    if model_name == 'fcomb':
        model = FCOMB(input_dim, num_fields, Config.evaluator_embedding_size, **model_params)
    elif model_name == 'pin':
        model_params['intersections'] = [[i, j] for i in range(num_fields) for j in range(i+1, num_fields)]
        model = FCOMB(input_dim, num_fields, Config.evaluator_embedding_size, **model_params)
    elif model_name == 'pnn':
        model_params['intersections'] = []
        model = FCOMB(input_dim, num_fields, Config.evaluator_embedding_size, **model_params)
    else:
        raise ValueError
    total_number_parameters = 0
    for var in tf.trainable_variables():
        number_parameters = 1
        for dim in var.get_shape():
            number_parameters *= dim.value
        total_number_parameters += number_parameters
    print("Model: {}\nNumber of parameters:{}\n\n".format(model_name, total_number_parameters))

    return model


class Model:
    def __init__(self, input_dim, num_fields, embed_size):
        self.inputs = None  # batch * num_fields
        self.labels = None  # batch
        self.embeded = None  # batch * num_fields * embed_size
        self.nn_inputs = None  # batch * nn_input_size
        self.logits = None  # batch
        self.loss = None  # batch

        self.input_dim = input_dim
        self.num_fields = num_fields
        self.embed_size = embed_size

    def define_placeholder(self):
        with tf.variable_scope("input"):
            self.inputs = tf.placeholder(tf.int32, shape=[None, self.num_fields], name="inputs")  # batch * input_dim
            self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")  # batch * 1

    def define_embedding(self):
        with tf.variable_scope("embedding"):
            initializer = get_initializer(init_type="xavier")
            w = tf.get_variable("w", shape=[self.input_dim, self.embed_size], dtype=tf.float32, initializer=initializer)
            self.embeded = tf.nn.embedding_lookup(w, self.inputs)  # batch * fields * embed_size

    def define_interaction(self, index, interaction, arch_type, comb_type):
        """
        :param interaction:
        :param arch_type in ["left", "middle"]
        :param comb_type in ["outer_product", "dense"]
        """
        assert arch_type in ["left", "middle"]
        assert comb_type in ["dense"]

        def dense(index, x, y):
            z = x * y
            input = tf.concat([x, y, z], axis=1)
            w1 = tf.get_variable(name="w_{}_1".format(index), shape=[input.shape.as_list()[-1], self.embed_size], dtype=tf.float32)
            b1 = tf.get_variable(name="b_{}_1".format(index), shape=[1, self.embed_size], dtype=tf.float32)
            w2 = tf.get_variable(name="w_{}_2".format(index), shape=[self.embed_size, self.embed_size], dtype=tf.float32)
            b2 = tf.get_variable(name="b_{}_2".format(index), shape=[1, self.embed_size], dtype=tf.float32)
            hide = tf.matmul(input, w1) + b1
            output = tf.matmul(hide, w2) + b2
            return output

        comb_map = {'dense': dense}

        g = comb_map.get(comb_type)

        with tf.variable_scope("interactions"):
            with tf.variable_scope("intersection_{}".format(index)):
                comb_index = 0
                if arch_type == 'left':
                    cur = self.embeded[:, interaction[0], :]
                    for i in range(1, len(interaction)):
                        cur = g(comb_index, cur, self.embeded[interaction[i]])
                        comb_index += 1
                    return cur
                elif arch_type == 'middle':
                    cur_inputs = [self.embeded[:, field, :] for field in interaction]
                    next_inputs = []
                    while len(cur_inputs) > 1:
                        for i in range(1, len(cur_inputs), 2):
                            next_inputs.append(g(comb_index, cur_inputs[i-1], cur_inputs[i]))
                            comb_index += 1
                        if len(cur_inputs) % 2 == 1:
                            next_inputs.append(cur_inputs[-1])
                        cur_inputs = next_inputs
                        next_inputs = []
                    return cur_inputs[0]

    def define_nn(self, layers):
        with tf.variable_scope("nn"):
            cur_layer = self.nn_inputs
            for index, (layer_dim, layer_act) in enumerate(layers):
                input_dim = cur_layer.shape.as_list()[-1]
                w = tf.get_variable("w_{}".format(index), shape=[input_dim, layer_dim], dtype=tf.float32)
                b = tf.get_variable("b_{}".format(index), shape=[1, layer_dim], dtype=tf.float32)
                cur_layer = tf.matmul(cur_layer, w) + b
                if layer_act is not None:
                    cur_layer = get_activation(layer_act)(cur_layer)

            self.logits = tf.reshape(cur_layer, shape=[-1])

    def define_loss(self):
        with tf.variable_scope("loss"):
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)


class FCOMB(Model):
    def __init__(self, input_dim, num_fields, embed_size, intersections, nn_layers):
        super(FCOMB, self).__init__(input_dim, num_fields, embed_size)

        self.define_placeholder()

        self.define_embedding()

        middles = [tf.reshape(self.embeded, shape=[-1, self.num_fields * self.embed_size])]
        for index, intersection in enumerate(intersections):
            middles.append(self.define_interaction(index, intersection, 'middle', 'dense'))

        self.nn_inputs = tf.concat(middles, axis=1)

        self.define_nn(nn_layers)

        self.define_loss()



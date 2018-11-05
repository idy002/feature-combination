import tensorflow as tf

WEIGHTS = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS]
EMBEDS = WEIGHTS + ["EMBEDS"]
NN_WEIGHTS = WEIGHTS + ["NN_WEIGHTS"]
SUB_NN_WEIGHTS = WEIGHTS + ["SUB_NN_WEIGHTS"]
BIASES = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES]


def as_model(model_name, **model_param):
    model_name = model_name.lower()
    model_map = {
        "lr": LR,
        "pin": PIN
    }
    return model_map[model_name](**model_param)


def get_activation(act_name):
    act_name = act_name.lower()
    act_map = {
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid,
        'tanh': tf.nn.tanh,
        'softmax': tf.nn.softmax,
        'elu': tf.nn.elu,
        'selu': tf.nn.selu,
        'crelu': tf.nn.crelu,
        'leacky_relu': tf.nn.leaky_relu,
        None: lambda x: x
    }
    if act_name in act_map:
        return act_map[act_name]
    else:
        raise ValueError


def get_initializer(init_type, minval=-0.001, maxval=0.001, mean=0, stddev=0.001, gain=1.):
    if type(init_type) is str:
        init_type = init_type.lower()
    assert init_type in {'xavier', 'orth', 'uniform', 'normal'} if type(init_type) is str \
        else type(init_type) in {int, float}, 'init type: {"xavier", "orth", "uniform", "normal", int, float}'
    if init_type == 'xavier':
        return tf.contrib.layers.xavier_initializer(uniform=True)
    elif init_type == 'orth':
        return tf.orthogonal_initializer(gain=gain)
    elif init_type == 'identity':
        return tf.initializers.identity(gain=gain)
    elif init_type == 'uniform':
        return tf.random_uniform_initializer(minval=minval, maxval=maxval)
    elif init_type == 'normal':
        return tf.truncated_normal_initializer(mean=mean, stddev=stddev)
    elif type(init_type) is int:
        return tf.constant_initializer(value=init_type, dtype=tf.int32)
    else:
        return tf.constant_initializer(value=init_type, dtype=tf.float32)


class Model:
    def __init__(self, input_dim, num_fields, output_dim=1):
        self.inputs = None
        self.training = None
        self.labels = None
        self.logits = None
        self.preds = None
        self.loss = None
        self.log_loss = None
        self.l2_loss = None

        self.embed = None  # batch * fields * embed_size
        self.pair = None  # batch * pair * (2 or 3) embed_size
        self.sub_nn_inputs = None  # batch * pair * a
        self.sub_nn_outputs = None  # batch * pair * b
        self.nn_inputs = None  # batch * c
        self.nn_outputs = None  # batch * d

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_fields = num_fields
        pass

    '''
    define self.inputs, self.labels and self.training
    '''

    def define_placeholder(self):
        with tf.variable_scope("input"):
            self.inputs = tf.placeholder(tf.int32, shape=[None, self.input_dim], name="inputs")  # batch * input_dim
            self.labels = tf.placeholder(tf.float32, shape=[None, self.output_dim], name="labels")  # batch * 1
            self.training = tf.placeholder(tf.bool, name="training")

    '''
    self.inputs --> self.embed
    '''

    def define_embedding(self, embed_size):
        with tf.variable_scope("embedding"):
            initializer = get_initializer(init_type="xavier")
            w = tf.get_variable("w", shape=[self.input_dim, embed_size], dtype=tf.float32, initializer=initializer, collections=EMBEDS)
            self.embed = tf.nn.embedding_lookup(w, self.inputs)  # batch * fields * embed_size

    '''
    self.embed --> self.pair
    '''

    def define_unroll(self, product_flag):
        p_indices = []
        q_indices = []
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                p_indices.append(i)
                q_indices.append(j)
        with tf.variable_scope("unroll"):
            a1 = tf.transpose(self.embed, [1, 0, 2])  # fields * batch * embed_size
            a2 = tf.gather(a1, p_indices)  # pair * batch * embed_size
            p_pair = tf.transpose(a2, [1, 0, 2])  # batch * pair * embed_size
            b1 = tf.transpose(self.embed, [1, 0, 2])
            b2 = tf.gather(b1, q_indices)
            q_pair = tf.transpose(b2, [1, 0, 2])
            if product_flag:
                self.pair = tf.concat([p_pair, q_pair, p_pair * q_pair], axis=2)  # batch * pair * 2 embed_size
            else:
                self.pair = tf.concat([p_pair, q_pair], axis=2)  # batch * pair * 3 embed_size

    '''
    self.sub_nn_inputs --> self.sub_nn_outputs
    '''

    def define_sub_nn(self, sub_nn_layers):
        with tf.variable_scope("sub_nn"):
            cur_dim = self.sub_nn_inputs.get_shape().as_list()[-1]
            cur_num = self.sub_nn_inputs.get_shape().as_list()[-2]
            cur_layer = tf.transpose(self.sub_nn_inputs, [1, 0, 2])  # cur_num * batch * cur_dim
            layer_index = 0
            for sl_type, sl_param in sub_nn_layers:
                if sl_type == "full":
                    new_dim = sl_param
                    with tf.variable_scope("layer_{}".format(layer_index)):
                        w = tf.get_variable("w", shape=[cur_num, cur_dim, new_dim], dtype=tf.float32,
                                            initializer=get_initializer("xavier"), collections=SUB_NN_WEIGHTS)
                        b = tf.get_variable("b", shape=[cur_num, 1, new_dim], dtype=tf.float32,
                                            initializer=get_initializer(init_type=0.0), collections=BIASES)
                        cur_layer = tf.matmul(cur_layer, w) + b
                    layer_index += 1
                    cur_dim = new_dim
                elif sl_type == "act":
                    cur_layer = get_activation(act_name=sl_param)(cur_layer)
                else:
                    print("sub_nn layer type must be in ['full', 'act'] but " + sl_type + " got")
                    raise ValueError
            self.sub_nn_outputs = tf.transpose(cur_layer, [1, 0, 2])

    '''
    self.nn_inputs --> self.nn_outputs
    '''

    def define_nn(self, nn_layers):
        with tf.variable_scope("nn"):
            cur_dim = self.nn_inputs.get_shape().as_list()[-1]
            cur_layer = self.nn_inputs  # batch * cur_dim
            layer_index = 0
            for l_type, l_param in nn_layers:
                if l_type == 'full':
                    new_dim = l_param
                    with tf.variable_scope("layer_{}".format(layer_index)):
                        w = tf.get_variable("w", shape=[cur_dim, new_dim], dtype=tf.float32, initializer=get_activation("xavier"),
                                            collections=NN_WEIGHTS)  # cur_dim * new_dim
                        b = tf.get_variable("b", shape=[1, new_dim], dtype=tf.float32, initializer=get_initializer(init_type=0.0),
                                            collections=BIASES)  # 1 * new_dim
                        cur_layer = tf.matmul(cur_layer, w) + b
                    layer_index += 1
                    cur_layer = new_dim
                elif l_type == 'act':
                    cur_layer = get_activation(act_name=l_param)(cur_layer)
                else:
                    print("nn layer type must be in ['full', 'act'] but " + l_type + " got")
                    raise ValueError
            self.nn_outputs = cur_layer

    '''
    self.logits --> self.log_loss, self.l2_loss, self.loss
    '''

    def define_loss(self, l2_embed=None, l2_subnn=None, l2_nn=None, l2_bias=None):
        with tf.variable_scope("loss"):
            l2_items = [(l2_embed, 'EMBEDS'), (l2_subnn, 'SUB_NN_WEIGHTS'), (l2_nn, 'NN_WEIGHTS'), (l2_bias, 'BIASES')]
            l2_losses = []
            for l2_scale, key in l2_items:
                if l2_scale is not None:
                    variables = tf.get_collection(key)
                    sub_l2_loss = tf.multiply(l2_scale, tf.add_n([tf.nn.l2_loss(v) for v in variables]), name=key)
                    l2_losses.append(sub_l2_loss)
            self.l2_loss = tf.add_n(l2_losses, name="l2_loss")
            self.log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits), name="log_loss")
            self.loss = tf.add(self.l2_loss, self.log_loss, name='loss')


class LR(Model):
    def __init__(self, input_dim, num_fields, output_dim=1):
        Model.__init__(input_dim, num_fields, output_dim)
        with tf.variable_scope("lr"):

            self.define_placeholder()

            self.define_embedding(1)

            bias = tf.get_variable("bias", shape=[1], dtype=tf.float32, collections=BIASES)

            self.logits = self.embed + bias

            self.define_loss(l2_embed=0.01, l2_bias=0.01)

class PIN(Model):
    def __init__(self, input_dim, num_fields, output_dim=1, embed_size=32):
        Model.__init__(input_dim, num_fields, output_dim)
        with tf.variable_scope("pin"):

            self.define_placeholder()

            self.define_embedding(embed_size=embed_size)

            self.define_unroll(product_flag=True)

            self.sub_nn_inputs = self.pair

            self.define_sub_nn([('full', 1)])

            sub_num = self.sub_nn_outputs.get_shape().as_list()[-2]
            sub_dim = self.sub_nn_outputs.get_shape().as_list()[-1]
            self.nn_inputs = tf.reshape(self.sub_nn_outputs, shape=[-1, sub_num * sub_dim])

            self.define_nn([('full', 400), ('act', 'relu'), ('full', 50), ('act', 'relu'), ('full', 1)])

            self.logits = self.nn_outputs

            self.define_loss()


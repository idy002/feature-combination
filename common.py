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


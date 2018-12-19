import os
import tensorflow.keras.backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # only show warning and error

from datasets import as_dataset
import tensorflow as tf


class Config:

    #
    #   general config
    #
    epoch_display_periods = 10  # epoch display periods
    summaries_dir = "./summaries"  # tensorboard writer target directory
    model_dir = "checkpoints"  # save model in this directory
    save_periods = 100  # save periods
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    keras_sess = tf.Session(config=sess_config)
    K.set_session(keras_sess)


    #
    #   environment config
    #
    environment_combination_len = 3
    environment_combinations_num = 10

    #
    #   actor config
    #
    lr = 0.001  # learning rate
    gamma = 0.5  # the discount factor in G
    value_scale = 0.5  # the weight of value function approximation in total loss
    reinforce_batch_size = 100  # batch size used in Reinforce algorithm
    gradient_clip = 40  # graient clip, avoid too large gradient

    #
    #   encoder config
    #
    encoder_dim = 64

    #
    #   reinforce config
    reinforce_logdir = "./summaries/reinforce_logdir"
    reinforce_learning_rate = 0.001


    #
    #   evaluator configs
    #
    evaluator_model_name = "lr"  #  'pin', 'lr'
    evaluator_optimizer_name = 'adam'
    evaluator_learning_rate = 0.03
    evaluator_epsilon = 1e-4
    evaluator_max_rounds = 2000
    evaluator_early_stop = 8
    evaluator_embedding_size = 20
    evaluator_log_step_frequency = 0
    evaluator_eval_round_frequency = 1
    evaluator_train_logdir = "./summaries/evaluator_train"
    evaluator_valid_logdir = "./summaries/evaluator_valid"
    evaluator_graph_logdir = "./summaries/evaluator_graph"


    #
    #   dataset
    #
    data_name = "ml1m"
    dataset = as_dataset(data_name, True)
    dataset.load_data(gen_type='train')
    dataset.load_data(gen_type='test')
    dataset.summary()
    num_fields = dataset.num_fields
    feat_sizes = dataset.feat_sizes
    feat_min = dataset.feat_min
    target_combination_num = 30
    target_combination_len = 4

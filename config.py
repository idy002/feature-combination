import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
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

    #
    #   environment config
    #
    environment_combination_len = 5
    environment_combinations_num = 100

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
    encoder_dim = 1024

    #
    #   reinforce config
    reinforce_logdir = "./summaries/reinforce_graph"
    reinforce_learning_rate = 0.001


    #
    #   evaluator configs
    #
    evaluator_model_name = "pin"
    evaluator_optimizer_name = 'adam'
    evaluator_learning_rate = 0.01
    evaluator_epsilon = 1e-5
    evaluator_train_logdir = "./summaries/evaluator_train"
    evaluator_valid_logdir = "./summaries/evaluator_valid"
    evaluator_graph_logdir = "./summaries/evaluator_graph"


    #
    #   dataset
    #
    data_name = "Couple"
    dataset = as_dataset(data_name, False)
    dataset.summary()
    num_fields = dataset.num_fields
    meta = dataset.meta
    target_combination_num = len(meta["field_combinations"])
    target_combination_len = meta["lens_fc"][0]

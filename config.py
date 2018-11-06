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

    #
    #   evaluator configs
    #
    evaluator_model_name = "lr"
    evaluator_optimizer_name = 'adam'
    evaluator_learning_rate = 0.001
    evaluator_epsilon = 1e-6
    evaluator_train_logdir = "./summaries/evaluator_train"
    evaluator_valid_logdir = "./summaries/evaluator_valid"


    #
    #   dataset
    #
    data_name = "Couple"
    dataset = as_dataset(data_name, False)
    num_fields = dataset.num_fields
    meta = dataset.meta
    target_combination_num = len(meta["field_combinations"])
    target_combination_len = meta["lens_fc"][0]

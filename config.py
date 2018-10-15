_data_path = "/home/yyding/Repositories/Ads-RecSys-Datasets"
_data_name = "Couple"
import sys
sys.path.append(_data_path)
from datasets import as_dataset


class Config:
    _dataset = as_dataset(_data_name, False)

    data_path = _data_path
    data_name = _data_name
    num_fields = _dataset.num_fields
    target_num_fields = 2
    #   learning rate
    lr = 0.001
    #   the discount factor in G
    gamma = 0.5
    #   the weight of value function approximation in total loss
    value_scale = 0.5
    #   batch size used in Reinforce algorithm
    reinforce_batch_size = 3
    #   tensorboard writer target directory
    summaries_dir = "./summaries"
    #   save model in this directory
    model_dir = "./checkpoints"
    # save periods
    save_periods = 100

    gradient_clip = 40

    #   epoch display periods
    epoch_display_periods = 10

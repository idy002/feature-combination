from config import Config
import sys
sys.path.append(Config.data_path)
from datasets import as_dataset

#
#   compute the reward of a filed combination
#     - state: one hot state that express the current fields combination
#     - action: the new field to be added
#   return:
#     - (state2, reward): state2 is the new field and reward is the score that the new state get
#
def get_reward(state, action):
    pass



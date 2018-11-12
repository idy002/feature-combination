import tensorflow as tf
import numpy as np
from environment.evaluator import Evaluator

from config import Config


class Enviroment:
    def __init__(self):
        self.evaluator = Evaluator()
        pass

    '''
    @:return hasStop, next_state, reward
    '''
    def do_step(self, state, action):
        fix_combs, cur_comb = state
        if np.sum(cur_comb) == Config.environment_combination_len - 1:
            cur_comb[action] = 1
            fix_combs = np.concatenate([fix_combs, cur_comb[np.newaxis, :]])
            cur_comb = np.zeros_like(cur_comb)
        next_state = (fix_combs, cur_comb)
        if np.sum(cur_comb) == 0:
            reward = self.evaluator.score(next_state, False)
        else:
            reward = 0.0
        hasStop = (fix_combs.shape[0] >= Config.environment_combinations_num)
        return hasStop, next_state, reward





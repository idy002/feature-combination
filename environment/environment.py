import tensorflow as tf
import numpy as np
from environment.old_evaluator import Evaluator
from learner.actor import State

from config import Config


class Enviroment:
    def __init__(self):
        self.evaluator = Evaluator()
        pass

    def reset(self):
        num_fields = Config.num_fields
        fix_combs = np.zeros(num_fields, dtype=np.int32)[np.newaxis, :]
        cur_comb = np.zeros(num_fields, dtype=np.int32)
        return State(fix_combs, cur_comb)

    @staticmethod
    def do_action(cur_combination, action):
        zero_index = -1
        for i in range(Config.num_fields):
            if cur_combination[i] == 0:
                zero_index += 1
                if zero_index == action:
                    cur_combination[i] = 1
                    return


    def step(self, state, action):
        """
        @:return hasStop, next_state, reward
        """
        fix_combs, cur_comb = np.array(state.fix_combinations), np.array(state.cur_combination)

        Enviroment.do_action(cur_comb, action)
        if np.sum(cur_comb) == Config.environment_combination_len:
            fix_combs = np.concatenate([fix_combs, cur_comb[np.newaxis, :]])
            cur_comb = np.zeros_like(cur_comb)
        next_state = State(fix_combs, cur_comb)
        if np.sum(cur_comb) == 0:
            new_score, auc = self.evaluator.score(next_state.fix_combinations[1:], False)
            old_score, _ = self.evaluator.score(next_state.fix_combinations[1:-1], False)
            reward = new_score - old_score
        else:
            reward, auc = 0.0, 0.0
        hasStop = (fix_combs.shape[0] - 1 >= Config.environment_combinations_num)
        if hasStop:
            self.report(next_state.fix_combinations)
        return hasStop, next_state, reward, auc

    def report(self, fix_combs):
        combs = [np.where(fix_combs[i])[0].tolist() for i in range(1, fix_combs.shape[0])]






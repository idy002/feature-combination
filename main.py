import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # only show warning and error

#from old_reinforce import Reinforce
from learner.reinforce import Reinforce
from environment.environment import Enviroment
from config import Config



env = Enviroment()

#  fix_combinations = np.zeros(shape=[len(Config.meta["field_combinations"]), Config.num_fields], dtype=np.int32)
#  for i_comb, comb in enumerate(Config.meta["field_combinations"]):
#      for a in comb:
#          fix_combinations[i_comb, a] = 1
#  print(fix_combinations)
#  print(env.evaluator.score(fix_combinations, True))
#
#  exit(0)

reinforce = Reinforce(learning_rate=0.01)
reinforce.train(env, num_batches=50, batch_size=3, discount_factor=1.0)


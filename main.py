import os

#from old_reinforce import Reinforce
from learner.reinforce import Reinforce
from environment.environment import Enviroment

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # only show warning and error

env = Enviroment()
reinforce = Reinforce(learning_rate=0.01)
reinforce.train(env, num_batches=10, batch_size=3, discount_factor=1.0)


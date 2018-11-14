import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # only show warning and error

#from old_reinforce import Reinforce
from learner.reinforce import Reinforce
from environment.environment import Enviroment


env = Enviroment()
reinforce = Reinforce(learning_rate=0.01)
reinforce.train(env, num_batches=50, batch_size=3, discount_factor=1.0)


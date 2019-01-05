import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # only show warning and error

# from old_reinforce import Reinforce
from learner.reinforce import Reinforce
from environment.env import Enviroment
from config import Config


def main():
    env = Enviroment()
    reinforce = Reinforce(learning_rate=0.001)
    reinforce.train(env, num_batches=5000, batch_size=3, discount_factor=1.0)

if __name__ == "__main__":
    main()

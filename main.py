import os

from old_reinforce import Reinforce

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # only show warning and error

reinforce = Reinforce()
reinforce.train(1000)

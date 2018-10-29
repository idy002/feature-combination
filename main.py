import os
from reinforce import Reinforce

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

reinforce = Reinforce()
reinforce.train(1000)



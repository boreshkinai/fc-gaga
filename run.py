# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
print("Tensorflow version:", tf.__version__)
from dataset import Dataset
from model import Trainer, Parameters
from model import hyperparams_defaults as hyperparams_dict

LOGDIR = "./logs"
DATADIR = "./data"

print("*********************************")
print("Default FC-GAGA parameters:")
print(hyperparams_dict)
print("*********************************")

hyperparams_dict["dataset"] = 'metr-la'
hyperparams_dict["horizon"] = 12
hyperparams_dict["history_length"] = 12

dataset = Dataset(name=hyperparams_dict["dataset"], 
                  horizon=hyperparams_dict["horizon"], 
                  history_length=hyperparams_dict["history_length"],
                  path=DATADIR)

hyperparams_dict["num_nodes"] = dataset.num_nodes
hyperparams = Parameters(**hyperparams_dict)

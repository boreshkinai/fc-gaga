# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
print("Tensorflow version:", tf.__version__)
from dataset import Dataset
from model import Trainer, Parameters
from model import hyperparams_defaults as hyperparams_dict

LOGDIR = "./logs"
DATADIR = "./data"


def insert_dict(d, k, v):
    previous = d.get(k, [])
    d[k] = previous + [v]
    return d

  
print("*********************************")
print("Default FC-GAGA parameters:")
print(hyperparams_dict)
print("*********************************")

hyperparams_dict["dataset"] = 'metr-la'
hyperparams_dict["horizon"] = 12
hyperparams_dict["history_length"] = 12

print("*********************************")
print("LOADING DATA")
print("*********************************")

dataset = Dataset(name=hyperparams_dict["dataset"], 
                  horizon=hyperparams_dict["horizon"], 
                  history_length=hyperparams_dict["history_length"],
                  path=DATADIR)

hyperparams_dict["num_nodes"] = dataset.num_nodes
hyperparams = Parameters(**hyperparams_dict)

print("*********************************")
print("TRAINING MODELS")
print("*********************************")

trainer = Trainer(hyperparams=hyperparams, logdir=LOGDIR)
trainer.fit(dataset=dataset)

print("*********************************")
print("COMPUTING METRICS")
print("*********************************")

early_stop_mae_h_repeats = dict()
early_stop_mape_h_repeats = dict()
early_stop_rmse_h_repeats = dict()
early_stop_mae_h_ave = dict()
early_stop_mape_h_ave = dict()
early_stop_rmse_h_ave = dict()
for i, h in enumerate(trainer.history):
    early_stop_idx = np.argmin(h['mae_val'])
    early_stop_mae = np.round(h['mae_test'][early_stop_idx], decimals=3)
    print(f"Early stop test error model {trainer.folder_names[i]}:", "Avg MAE", early_stop_mae)
    for horizon in range(3, hyperparams.horizon+1, 3):

        early_stop_mae_h_repeats = insert_dict(early_stop_mae_h_repeats, k=horizon, v=h[f'mae_test_h{horizon}'][early_stop_idx])
        early_stop_mape_h_repeats = insert_dict(early_stop_mape_h_repeats, k=horizon, v=h[f'mape_test_h{horizon}'][early_stop_idx])
        early_stop_rmse_h_repeats = insert_dict(early_stop_rmse_h_repeats, k=horizon, v=h[f'rmse_test_h{horizon}'][early_stop_idx])
        
        print(f"Horizon {horizon} MAE:", np.round(early_stop_mae_h_repeats[horizon][-1], decimals=2), 
              f"Horizon {horizon} MAPE:", np.round(early_stop_mape_h_repeats[horizon][-1], decimals=2), 
              f"Horizon {horizon} RMSE:", np.round(early_stop_rmse_h_repeats[horizon][-1], decimals=2))

    for horizon in range(3, hyperparams.horizon+1, 3):
        early_stop_mae_h_ave[horizon] = np.round(np.mean(early_stop_mae_h_repeats[horizon]), decimals=2)
        early_stop_mape_h_ave[horizon] = np.round(np.mean(early_stop_mape_h_repeats[horizon]), decimals=2)
        early_stop_rmse_h_ave[horizon] = np.round(np.mean(early_stop_rmse_h_repeats[horizon]), decimals=2)

print()
print("Average MAE:", early_stop_mae_h_ave)
print("Average MAPE:", early_stop_mape_h_ave)
print("Average RMSE:", early_stop_rmse_h_ave)

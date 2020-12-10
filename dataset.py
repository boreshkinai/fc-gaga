import numpy as np
import os
import time
from google_drive_downloader import GoogleDriveDownloader as gdd
import pathlib
from generate_training_data import generate_train_val_test
from typing import NamedTuple


class DatasetParameters(NamedTuple):
    history_length: int
    horizon: int
    traffic_df_filename: str
    output_dir: str


class Dataset(object):
    def __init__(self, name='metr-la', horizon: int = 3, history_length: int = 3, path: str = 'data'):
        self.horizon = horizon
        self.history_length = history_length
        self.name = name

        if self.name == 'metr-la':
            gdd.download_file_from_google_drive(file_id='1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC', dest_path=f"{path}/metr-la.h5", unzip=False)
        elif self.name == 'pems-bay':
            gdd.download_file_from_google_drive(file_id='1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq', dest_path=f"{path}/pems-bay.h5", unzip=False)
        else:
            raise Exception(f"unknown dataset: {self.name}")
        pathlib.Path(f"{path}/{self.name}").mkdir(parents=True, exist_ok=True)

        dataset_parameters = {"history_length": self.history_length, 
                              "horizon": self.horizon, 
                              "traffic_df_filename": f"{path}/{self.name}.h5",
                              "output_dir": f"{path}/{self.name}"}

        self.data = {}
        for category in ['train', 'val', 'test']:
            data_filename = os.path.join(dataset_parameters["output_dir"], category + f"-history-{self.history_length}-horizon-{self.horizon}.npz")
            if not os.path.isfile(data_filename):
                generate_train_val_test(DatasetParameters(**dataset_parameters))
            cat_data = np.load(data_filename)
            self.data['x_' + category] = np.float32(cat_data['x'])
            self.data['y_' + category] = np.float32(cat_data['y'])
                
        self.num_nodes = self.data['x_train'].shape[-2]
        for category in ['train', 'val', 'test']:
            self.data['x_' + category] = np.transpose(self.data['x_' + category], (0, 2, 1, 3))
            self.data['y_' + category] = np.transpose(self.data['y_' + category], (0, 2, 1, 3))

    def get_batch(self, batch_size: int = 1024):
        ts_idxs = np.random.choice(np.arange(len(self.data['x_train'])), size=batch_size, replace=True)
        ids = np.tile(np.arange(self.num_nodes)[np.newaxis,:], reps=[batch_size,1])
        batch = dict()
        batch['x'] = self.data['x_train'][ts_idxs]
        batch['y'] = self.data['y_train'][ts_idxs][...,0]
        batch['node_id'] = ids
        return batch

    def get_sequential_batch(self, batch_size: int = 1000, split: str = 'test'):
        num_batches = int(np.ceil(len(self.data[f"x_{split}"]) / batch_size))
        for i in range(num_batches):
            ts_idxs = range(i*batch_size, min((i+1)*batch_size, len(self.data[f"x_{split}"])))
            ids = np.tile(np.arange(self.num_nodes)[np.newaxis,:], reps=[batch_size,1])
            batch = dict()
            batch['x'] = self.data[f"x_{split}"][ts_idxs]
            batch['y'] = self.data[f"y_{split}"][ts_idxs][...,0]
            batch['node_id'] = ids
            yield batch


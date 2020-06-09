import tensorflow as tf
import numpy as np
import os
from itertools import product
from typing import Dict, NamedTuple, Union
from utils import MetricsCallback


hyperparams_defaults = {
    "dataset": "metr-la", 
    "repeat": list(range(5)),
    "epochs": [60], 
    "steps_per_epoch": [800],  # 800 METR-LA, 800 PEMS-BAY
    "block_layers": 3,
    "hidden_units": 128,
    "blocks": 2,
    "horizon": 12,
    "history_length": 12,
    "init_learning_rate": 1e-3,
    "decay_steps": 3, 
    "decay_rate": 0.5,
    "batch_size": 4,
    "weight_decay": 1e-5,
    "node_id_dim": 64,
    "num_nodes": 207, # 207 | 325
    "num_stacks": [3],
    "epsilon": 10
}

class Parameters(NamedTuple):
    dataset: str
    repeat: int
    epochs: int
    steps_per_epoch: int
    block_layers: int
    hidden_units: int
    blocks: int
    horizon: int
    history_length: int
    init_learning_rate: float
    decay_steps: int
    decay_rate: float
    batch_size: int
    weight_decay: float
    node_id_dim: int
    num_nodes: int
    num_stacks: int
    epsilon: float


class FcBlock(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, input_size:int, output_size: int, **kw):
        super(FcBlock, self).__init__(**kw)
        self.hyperparams = hyperparams
        self.input_size = input_size
        self.output_size = output_size
        self.fc_layers = []
        for i in range(hyperparams.block_layers):
            self.fc_layers.append(
                tf.keras.layers.Dense(hyperparams.hidden_units, 
                                      activation=tf.nn.relu,
                                      kernel_regularizer=tf.keras.regularizers.l2(hyperparams.weight_decay),
                                      name=f"fc_{i}")
            )
        self.forecast = tf.keras.layers.Dense(self.output_size, activation=None, name="forecast")
        self.backcast = tf.keras.layers.Dense(self.input_size, activation=None, name="backcast")   
        
    def call(self, inputs, training=False):
        h = self.fc_layers[0](inputs)
        for i in range(1, self.hyperparams.block_layers):
            h = self.fc_layers[i](h)
        backcast = tf.keras.activations.relu(inputs - self.backcast(h))
        return backcast, self.forecast(h)


class FcGagaLayer(tf.keras.layers.Layer):
    def __init__(self, hyperparams: Parameters, input_size:int, output_size: int, num_nodes:int, **kw):
        super(FcGagaLayer, self).__init__(**kw)
        self.hyperparams = hyperparams
        self.num_nodes = num_nodes
        self.input_size = input_size
        
        self.blocks = []
        for i in range(self.hyperparams.blocks): 
            self.blocks.append(FcBlock(hyperparams=hyperparams, 
                                           input_size=self.input_size, 
                                           output_size=hyperparams.horizon, 
                                           name=f"block_{i}"))
                
        self.node_id_em = tf.keras.layers.Embedding(input_dim=self.num_nodes, 
                                                    output_dim=self.hyperparams.node_id_dim, 
                                                    embeddings_initializer='uniform',
                                                    input_length=self.num_nodes, name="dept_id_em",
                                                    embeddings_regularizer=tf.keras.regularizers.l2(hyperparams.weight_decay))  
        
        self.time_gate1 = tf.keras.layers.Dense(hyperparams.hidden_units, 
                                               activation=tf.keras.activations.relu,
                                               name=f"time_gate1")
        self.time_gate2 = tf.keras.layers.Dense(hyperparams.horizon, 
                                               activation=None,
                                               name=f"time_gate2")
        self.time_gate3 = tf.keras.layers.Dense(hyperparams.history_length, 
                                               activation=None,
                                               name=f"time_gate3")
        
    def call(self, history_in, node_id_in, time_of_day_in, training=False):
        node_id = self.node_id_em(node_id_in)

        node_embeddings = tf.squeeze(node_id[0,:,:])
        node_id = tf.squeeze(node_id, axis=-2)

        time_gate = self.time_gate1(tf.concat([node_id, time_of_day_in], axis=-1))
        time_gate_forward = self.time_gate2(time_gate)
        time_gate_backward = self.time_gate3(time_gate)

        history_in = history_in / (1.0 + time_gate_backward)

        node_embeddings_dp = tf.tensordot(node_embeddings,  tf.transpose(node_embeddings, perm=[1,0]), axes=1)
        node_embeddings_dp = tf.math.exp(self.hyperparams.epsilon*node_embeddings_dp)
        node_embeddings_dp = node_embeddings_dp[tf.newaxis,:,:,tf.newaxis]

        level = tf.reduce_max(history_in, axis=-1, keepdims=True) 

        history = tf.math.divide_no_nan(history_in, level)
        # Add history of all other nodes
        shape = history_in.get_shape().as_list()
        all_node_history = tf.tile(history_in[:,tf.newaxis,:,:], multiples=[1,self.num_nodes,1,1])

        all_node_history = all_node_history * node_embeddings_dp
        all_node_history = tf.reshape(all_node_history, shape=[-1, self.num_nodes, self.num_nodes*shape[2]])
        all_node_history = tf.math.divide_no_nan(all_node_history - level, level)
        all_node_history = tf.where(all_node_history > 0, all_node_history, 0.0) 
        history = tf.concat([history, all_node_history], axis=-1)
        # Add node ID
        history = tf.concat([history, node_id], axis=-1)

        backcast, forecast_out = self.blocks[0](history)
        for i in range(1, self.hyperparams.blocks):
            backcast, forecast_block = self.blocks[i](backcast)
            forecast_out = forecast_out + forecast_block
        forecast_out = forecast_out[:,:,:self.hyperparams.horizon]
        forecast = forecast_out * level

        forecast = (1.0 + time_gate_forward) * forecast

        return backcast, forecast


class FcGaga:
    def __init__(self, hyperparams: Parameters, name: str='fcgaga', logdir: str='logs', num_nodes: int = 100):
        super(FcGaga, self).__init__()
        self.hyperparams = hyperparams
        self.name=name
        self.logdir=logdir
        self.num_nodes = num_nodes
        self.input_size = self.hyperparams.history_length + self.hyperparams.node_id_dim + self.num_nodes*self.hyperparams.history_length

        self.fcgaga_layers = []
        for i in range(hyperparams.num_stacks):
            self.fcgaga_layers.append(FcGagaLayer(hyperparams=hyperparams,
                                                  input_size=self.input_size, 
                                                  output_size=hyperparams.horizon, 
                                                  num_nodes = self.num_nodes, 
                                                  name=f"fcgaga_{i}")
                                      )

        inputs, outputs = self.get_model()
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        self.inputs = inputs
        self.inputs = outputs
        self.model = model
                
    def get_model(self):
        history_in = tf.keras.layers.Input(shape=(self.num_nodes, self.hyperparams.history_length), name='history')
        time_of_day_in = tf.keras.layers.Input(shape=(self.num_nodes, self.hyperparams.history_length), name='time_of_day')
        node_id_in = tf.keras.layers.Input(shape=(self.num_nodes, 1), dtype=tf.uint16, name='node_id')
        
        backcast, forecast = self.fcgaga_layers[0](history_in=history_in, node_id_in=node_id_in, time_of_day_in=time_of_day_in)
        for nbg in self.fcgaga_layers[1:]:
            backcast, forecast_graph = nbg(history_in=forecast, node_id_in=node_id_in, time_of_day_in=time_of_day_in)
            forecast = forecast + forecast_graph
        forecast = forecast / self.hyperparams.num_stacks
        forecast = tf.where(tf.math.is_nan(forecast), tf.zeros_like(forecast), forecast)

        inputs = {'history': history_in, 'node_id': node_id_in, 
                  'time_of_day': time_of_day_in} 
        outputs = {'targets': forecast}
        return inputs, outputs


class Trainer:
    def __init__(self, hyperparams: Parameters, logdir: str):
        inp = dict(hyperparams._asdict())
        values = [v if isinstance(v, list) else [v] for v in inp.values()]
        self.hyperparams = [Parameters(**dict(zip(inp.keys(), v))) for v in product(*values)]
        inp_lists = {k: v  for k, v in inp.items() if isinstance(v, list)}
        values = [v for v in inp_lists.values()]
        variable_values = [dict(zip(inp_lists.keys(), v)) for v in product(*values)]
        folder_names = []
        for d in variable_values: 
            folder_names.append(
                ';'.join(['%s=%s' % (key, value) for (key, value) in d.items()])
            )
        self.history = []
        self.forecasts = []
        self.models = []
        self.logdir = logdir
        self.folder_names = folder_names
        for i, h in enumerate(self.hyperparams): 
            self.models.append(FcGaga(hyperparams=h, name=f"fcgaga_model_{i}", 
                                      logdir=os.path.join(self.logdir, folder_names[i]),
                                      num_nodes=h.num_nodes))
            
    def generator(self, ds, hyperparams: Parameters):
        while True:
            batch = ds.get_batch(batch_size=hyperparams.batch_size)
            weights = np.all(batch["y"] > 0, axis=-1, keepdims=False).astype(np.float32)
            weights = weights / np.prod(weights.shape)
            yield  {"history": batch["x"][...,0], "node_id": batch["node_id"], "time_of_day": batch["x"][...,1]}, \
                   {"targets": batch["y"]}, \
                   weights                  
        
    def fit(self, dataset, verbose=1):
        for i, hyperparams in enumerate(self.hyperparams):
            if verbose > 0:
                print(f"Fitting model {i+1} out of {len(self.hyperparams)}, {self.folder_names[i]}")
            
            boundary_step = hyperparams.epochs // 10
            boundary_start = hyperparams.epochs - boundary_step*hyperparams.decay_steps - 1
            
            boundaries = list(range(boundary_start, hyperparams.epochs, boundary_step))
            values = list(hyperparams.init_learning_rate * hyperparams.decay_rate ** np.arange(0, len(boundaries)+1))
            scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values)
            
            lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

            metrics = MetricsCallback(dataset=dataset, logdir=self.models[i].logdir)
            tb = tf.keras.callbacks.TensorBoard(log_dir=self.models[i].logdir, embeddings_freq=10)
            
            self.models[i].model.compile(optimizer=tf.keras.optimizers.Adam(),
                                         loss={"targets": tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)},
                                         loss_weights={"targets": 1.0})
                        
            fit_output = self.models[i].model.fit(self.generator(ds=dataset, hyperparams=hyperparams),
                                            callbacks=[lr, metrics], # tb
                                            epochs=hyperparams.epochs, 
                                            steps_per_epoch=hyperparams.steps_per_epoch, 
                                            verbose=verbose)
            self.history.append(fit_output.history)
            
        
    
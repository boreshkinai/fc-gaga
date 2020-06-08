import numpy as np
import tensorflow as tf

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))

def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)

def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return 100*np.mean(mape)
    
def masked_mae_np(preds, labels, null_val=np.nan):

    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)
        
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, logdir):
        super().__init__()

        self.validation_data = dataset.get_sequential_batch(batch_size=len(dataset.data['x_val']),
                                                            split='val').__next__()
        self.test_data = dataset.get_sequential_batch(batch_size=len(dataset.data['x_test']),
                                                      split='test').__next__()

    def on_train_begin(self, logs={}):
        pass    

    def on_epoch_end(self, epoch, logs={}):
        prediction_val = self.model.predict({"history": self.validation_data["x"][...,0], 
                                             "node_id": self.validation_data["node_id"],
                                             "time_of_day": self.validation_data["x"][...,1]})
        
        prediction_test = self.model.predict({"history": self.test_data["x"][...,0], 
                                              "node_id": self.test_data["node_id"],
                                              "time_of_day": self.test_data["x"][...,1]})

        logs['mae_val'] = masked_mae_np(preds=prediction_val['targets'], labels=self.validation_data['y'], null_val=0)
        logs['mae_test'] = masked_mae_np(preds=prediction_test['targets'], labels=self.test_data['y'], null_val=0)

        for h in range(prediction_test['targets'].shape[-1]):
            logs[f'mae_val_h{h+1}'] = masked_mae_np(preds=prediction_val['targets'][...,h], labels=self.validation_data['y'][...,h], null_val=0)
            logs[f'mae_test_h{h+1}'] = masked_mae_np(preds=prediction_test['targets'][...,h], labels=self.test_data['y'][...,h], null_val=0)
            logs[f'mape_val_h{h+1}'] = masked_mape_np(preds=prediction_val['targets'][...,h], labels=self.validation_data['y'][...,h], null_val=0)
            logs[f'rmse_val_h{h+1}'] = masked_rmse_np(preds=prediction_val['targets'][...,h], labels=self.validation_data['y'][...,h], null_val=0)
            logs[f'mape_test_h{h+1}'] = masked_mape_np(preds=prediction_test['targets'][...,h], labels=self.test_data['y'][...,h], null_val=0)
            logs[f'rmse_test_h{h+1}'] = masked_rmse_np(preds=prediction_test['targets'][...,h], labels=self.test_data['y'][...,h], null_val=0)

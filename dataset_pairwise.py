import numpy as np
import os
import time


class PemsDataset(object):
    def __init__(self, path, batch_size, do_transform=False, speed_only=True, horizon=3, history_length=3):

        self.batch_size = batch_size
        self.do_transform = do_transform
        self.horizon = horizon
        self.history_length = history_length

        self.data = {}
        # self.data['x_test'] = np.arange(200, 1200).reshape(50, 10, 2)
        # self.data['y_test'] = np.arange(3000, 4000).reshape(50, 10, 2)

        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(path, category + f"-history-{self.history_length}-horizon-{self.horizon}.npz"))
            if speed_only:
                self.data['x_' + category] = np.float32(cat_data['x'][..., :1])
                self.data['y_' + category] = np.float32(cat_data['y'][..., :1])
            else:
                self.data['x_' + category] = np.float32(cat_data['x'])
                self.data['y_' + category] = np.float32(cat_data['y'])

        self.num_nodes = self.data['x_train'].shape[-2]

        if self.do_transform:
            self.mean = self.data['x_train'][...,0].mean()
            self.std = self.data['x_train'][...,0].std()
        else:
            self.mean = 0.0
            self.std = 1.0

        for category in ['train', 'val', 'test']:
            self.data['x_' + category][...,0] = self.transform(self.data['x_' + category][...,0])
            self.data['y_' + category][...,0] = self.transform(self.data['y_' + category][...,0])
            self.data['x_' + category] = np.transpose(self.data['x_' + category], (0, 2, 1, 3))
            self.data['y_' + category] = np.transpose(self.data['y_' + category], (0, 2, 1, 3))

    def get_batch(self, batch_size: int = 1024):
        ts_idxs = np.random.choice(np.arange(len(self.data['x_train'])), size=batch_size, replace=True)
        ids = np.tile(np.arange(self.num_nodes)[np.newaxis,:], reps=[batch_size,1])
        batch = dict()
        batch['x'] = self.data['x_train'][ts_idxs]
        batch['y'] = self.data['y_train'][ts_idxs][...,0]
        # batch['time_of_day'] = self.data[f"y_train"][ts_idxs][...,1]
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
            # batch['time_of_day'] = self.data[f"y_{split}"][ts_idxs][...,1]
            batch['node_id'] = ids
            yield batch

    def transform(self, data):
        return (data - self.mean) / self.std

    def untransform(self, data):
        return data*self.std + self.mean

    def pairwise(self, x):
        # x_pw = np.empty((x.shape[0], x.shape[0], x.shape[1] * 2))
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[0]):
        #         x_pw[i, j] = np.concatenate((x[i], x[j]))
        a = np.tile(x, (1, x.shape[0])).reshape(-1, x.shape[1])
        b = np.tile(x, (x.shape[0], 1)).reshape(-1, x.shape[1])
        x_pw = np.concatenate((a, b), axis=1).reshape(x.shape[0], x.shape[0], -1)

        return x_pw

    def get_iterator(self, state):
        # shuffle
        if state == 0:
            train_size = self.data['x_train'].shape[0]
            permutation = np.random.permutation(train_size)
            xs = self.data['x_train'][permutation]
            ys = self.data['y_train'][permutation]  # (52..., 325, 3)

        elif state == 1:
            xs = self.data['x_val']
            ys = self.data['y_val']

        else:
            xs = self.data['x_test']
            ys = self.data['y_test']

        def _wrapper():
            self.size = xs.shape[0]
            self.num_nodes = xs.shape[1]
            self.num_batch = int(np.ceil(self.num_nodes / self.batch_size))
            self.row_idx = 0
            while self.row_idx < self.size:
                self.node_idx = 0
                x = np.concatenate((np.arange(self.num_nodes)[:, None], xs[self.row_idx]), axis=1)
                y = ys[self.row_idx]
                if state == 0 and self.pw:
                    node_perm = np.random.permutation(self.num_nodes)
                    x, y = x[node_perm], y[node_perm]
                if self.pw:
                    x = self.pairwise(x)
                while self.node_idx < self.num_nodes:
                    end = min(self.node_idx + self.batch_size, self.num_nodes)
                    if self.pw:
                        yield x[self.node_idx:end], y[self.node_idx:end], [0, 0]
                    else:
                        yield np.expand_dims(x, axis=0), y[self.node_idx:end], [self.node_idx, end]
                    self.node_idx += self.batch_size
                self.row_idx += 1
        return _wrapper()


if __name__ == '__main__':
    train_loader = PemsDataset('METR-LA', 207, True)
    for epoch in range(1):
        # for i, (x, y) in enumerate(train_loader.get_iterator(0)):
        #     if i==0:
        #         # print('epoch', epoch + 1, 'batch', i + 1)
        #         # x = inverse_transform(x, mean, std)
        #         # y = inverse_transform(y, mean, std)
        #         print('train x', x, 'train y,',  y)
        #         print(x.shape, y.shape)
        gts_test = []
        start = time.time()
        c = 0
        for i, (x, y, rng) in enumerate(train_loader.get_iterator(0)):
            c += 1
            # print('epoch', epoch + 1, 'batch', i + 1)
            # print('train x', x, 'train y,', y)
            # gts_test.append(y)
            # print(x.shape, y.shape)
        print('Elapsed time:', time.time() - start, c)
    #     print(np.array_equal(train_loader.data['y_test'], np.concatenate(gts_test, axis=0).reshape(-1, 10, 2)))
    # groundtruth_test = np.concatenate(gts_test, axis=0)
    print('Done!')

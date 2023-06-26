from pathlib import Path
import torch
import h5py
import numpy as np
from .tensor_dataset import TensorDataset

def load_advection(data_path, n_train, n_test, batch_size=32, batch_test=100, time=1, grid=[0,1]):

    data_path = Path(data_path).joinpath('1D_Advection_Sols_beta0.1.hdf5').as_posix()
    with h5py.File(data_path, 'r') as f:
        keys = list(f.keys())
        keys.sort()

        data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...

        print('data shape', data.shape)
        data = data[:, :, ]
        data = data[:, :, :, None]
        data = np.transpose(data, (0, 3, 2, 1)) # datapoints, channel, x-spatial, time
        print('data shape', data.shape)

        # todo: set up grid / positional encoding
        #grid = np.array(f["x-coordinate"], dtype=np.float32)
        #grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)

        test_idx = 1000 # todo: don't hard code this

        x_test = torch.Tensor(data[:test_idx, :, :, 0])
        y_test = torch.Tensor(data[:test_idx, :, :, 1])
        x_train = torch.Tensor(data[test_idx:, :, :, 0])
        y_train = torch.Tensor(data[test_idx:, :, :, 1])

        train_db = TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_db,batch_size=batch_size, shuffle=False)

        test_db = TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_db,batch_size=batch_test, shuffle=False)

        # todo: add positional encoder, normalization, output encoder
        return train_loader, test_loader, None

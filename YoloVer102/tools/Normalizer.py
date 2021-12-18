import torch
import numpy as np


def generic_normalize(data: any, alpha=0., beta=1., gamma=1., float_type=True, tensor_output=True):
    # convert data to float type if float_type is True
    if float_type and isinstance(data, np.ndarray):
        data = data.astype(np.float32)
    elif float_type and isinstance(data, torch.Tensor):
        data = data.type(torch.FloatTensor).numpy()

    # normalize data
    data = (data - alpha) / beta * gamma

    # if float_type if False, convert the data to integer
    if not float_type and isinstance(data, np.ndarray):
        data = data.astype(np.int32)
    elif not float_type and isinstance(data, torch.Tensor):
        data = data.type(torch.IntTensor)

    # return tensor if return_tensor is True
    if tensor_output and isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        return data
    elif tensor_output and isinstance(data, torch.Tensor):
        return data
    elif not tensor_output and isinstance(data, torch.Tensor):
        return data.numpy()
    elif not tensor_output and isinstance(data, np.ndarray):
        return data
    else:
        return None


def standard_normalize(data: any, mu=0., sigma=1., float_type=True, tensor_output=True):
    # convert data to float type if float_type is True
    if float_type and isinstance(data, np.ndarray):
        data = data.astype(np.float32)
    elif float_type and isinstance(data, torch.Tensor):
        data = data.type(torch.FloatTensor).numpy()

    # normalize data
    data = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((data - mu) ** 2) / (2 * sigma ** 2))

    # if float_type if False, convert the data to integer
    if not float_type and isinstance(data, np.ndarray):
        data = data.astype(np.int32)
    elif not float_type and isinstance(data, torch.Tensor):
        data = data.type(torch.IntTensor)

    # return tensor if return_tensor is True
    if tensor_output and isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        return data
    elif tensor_output and isinstance(data, torch.Tensor):
        return data
    elif not tensor_output and isinstance(data, torch.Tensor):
        return data.numpy()
    elif not tensor_output and isinstance(data, np.ndarray):
        return data
    else:
        return None


def test():
    np2 = generic_normalize(np.array([0, 100, 200, 300]), 0, 300, 1)
    np3 = standard_normalize(np.array([0, 100, 200, 300]), 0., 0.3081)
    print(np2)
    print(np3)


if __name__ == '__main__':
    test()

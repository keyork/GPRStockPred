
import torch
import pandas as pd
import numpy as np
import pywt


def read_data(file_path):

    if file_path[-1] == 's' or file_path[-1] == 'x':
        df = pd.read_excel(file_path)
    elif file_path[-1] == 'v':
        df = pd.read_csv(file_path)
    else:
        return False
    pure_data = df.iloc[0:, 1:-1]
    
    np_temp_raw = np.array(pure_data, dtype=np.float32)
    np_temp = np_temp_raw[::-1].copy()
    
    all_data = torch.tensor(np_temp,dtype=torch.float32)

    return all_data


def get_data(all_data, train_scale):

    data_len = all_data.size()[0]
    train_len = int(data_len * train_scale)
    test_len = data_len - train_len

    train_x = torch.linspace(0, train_len-1, train_len)
    train_y = all_data[0:train_len, 3]

    # test_x = torch.linspace(train_len, data_len-1, test_len)
    test_x = torch.linspace(0, data_len-1, data_len)
    draw_test_x = torch.linspace(train_len, data_len-1, test_len)
    test_y = all_data[train_len:, 3]

    return train_x, train_y, test_x, test_y, draw_test_x


def res_data(all_data, train_scale):

    data_len = all_data.size()[0] - 1
    train_len = int(data_len * train_scale)
    test_len = data_len - train_len

    raw_y = all_data[:,3]
    roll_y = torch.roll(raw_y, 1)
    res_y = raw_y - roll_y
    res_y = res_y[1:]

    train_x = torch.linspace(0, train_len-1, train_len)
    train_y = res_y[:train_len]

    test_x = torch.linspace(0, data_len-1, data_len)
    draw_test_x = torch.linspace(train_len, data_len-1, test_len)
    test_y = res_y[train_len:]

    return train_x, train_y, test_x, test_y, draw_test_x, raw_y[0]


def dwt_data_ca(all_data, train_scale):

    data_len = all_data.size()[0]
    train_len = int(data_len * train_scale)
    test_len = data_len - train_len

    if train_len % 2 == 1:
        train_y = all_data[1:train_len, 3]
        train_len -= 1
    else:
        train_y = all_data[0:train_len, 3]
    
    if test_len % 2 == 1:
        test_y = all_data[train_len:-1, 3]
        test_len -= 1
    else:
        test_y = all_data[train_len:, 3]

    

    (train_ca, train_cd) = pywt.dwt(train_y.numpy(), 'db1')
    (test_ca, test_cd) = pywt.dwt(test_y.numpy(), 'db1')
    
    train_y = torch.from_numpy(train_ca)
    test_y = torch.from_numpy(test_ca)
    

    train_len = train_ca.shape[0]
    test_len = test_ca.shape[0]
    data_len = train_len + test_len


    train_x = torch.linspace(0, train_len-1, train_len)

    test_x = torch.linspace(0, data_len-1, data_len)
    draw_test_x = torch.linspace(train_len, data_len-1, test_len)

    return train_x, train_y, test_x, test_y, draw_test_x

def dwt_data_cd(all_data, train_scale):
    
    data_len = all_data.size()[0]
    train_len = int(data_len * train_scale)
    test_len = data_len - train_len

    if train_len % 2 == 1:
        train_y = all_data[1:train_len, 3]
        train_len -= 1
    else:
        train_y = all_data[0:train_len, 3]
    
    if test_len % 2 == 1:
        test_y = all_data[train_len:-1, 3]
        test_len -= 1
    else:
        test_y = all_data[train_len:, 3]

    

    (train_ca, train_cd) = pywt.dwt(train_y.numpy(), 'db1')
    (test_ca, test_cd) = pywt.dwt(test_y.numpy(), 'db1')
    
    train_y = torch.from_numpy(train_cd)
    test_y = torch.from_numpy(test_cd)
    

    train_len = train_cd.shape[0]
    test_len = test_cd.shape[0]
    data_len = train_len + test_len

    train_x = torch.linspace(0, train_len-1, train_len)

    test_x = torch.linspace(0, data_len-1, data_len)
    draw_test_x = torch.linspace(train_len, data_len-1, test_len)

    return train_x, train_y, test_x, test_y, draw_test_x
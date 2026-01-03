import json
import os
import pickle

import numpy as np


def read_dir(data_dir):
    data = []
    
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    clients = list(range(1,len(files)+1))
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        data.append(cdata)
    
    return clients, data


def read_data(train_data_dir, test_data_dir):
    pre_dir = os.path.split(train_data_dir)[0]
    cache_path = os.path.join(pre_dir, "data_cache.obj")
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            (train_clients, train_data, test_clients, test_data) = [pickle.load(f) for _ in range(4)]
    else:
        print('train_dir', train_data_dir)
        train_clients, train_data = read_dir(train_data_dir)
        test_clients, test_data = read_dir(test_data_dir)
        with open(cache_path, 'wb') as f:
            for data in (train_clients, train_data, test_clients, test_data):
                pickle.dump(data, f)
    assert sorted(train_clients) == sorted(test_clients)
    
    return train_clients, train_data, test_data

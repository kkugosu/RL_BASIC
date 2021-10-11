import numpy as np
import random

class custom_dataloder():

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_category = len(self.dataset[0])

    def __iter__(self):
        return self

    def __next__(self):
        index = random.sample(list(range(len(self.dataset))), self.batch_size)

        _batchdata = np.array(self.dataset[index[0]], dtype=np.object)
        _cnum = 0
        while _cnum < self.num_category:
            i = 0
            i = i + 1
            _batchdata[_cnum] = np.array([_batchdata[_cnum]])
            while i < self.batch_size:
                added_data = np.array([self.dataset[index[i]][_cnum]])
                _batchdata[_cnum] = np.concatenate((_batchdata[_cnum], added_data), axis=0)
                i = i + 1
            _cnum = _cnum + 1

        return _batchdata


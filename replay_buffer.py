import numpy as np

from collections import deque

import random


class ReplayBuffer(object):

    def __init__(self, maxlen, seed=None):
        self.maxlen = maxlen

        random.seed(seed)

        self.clear()

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = deque(maxlen=self.maxlen)

    def sample(self, list_, num):
        if len(list_) == 0:
            return []
        
        if len(list_) < num:
            return list(self.buffer)

        return random.sample(list_, k=num)

    def sample_buffer(self, num):
        return self.sample(self.buffer, num)

    def append(self, item):
        self.buffer.append(item)

    def extend(self, items):
        self.buffer.extend(items)

    def pop(self):
        return self.buffer.pop()

    def pop_and_append(self):
        item = self.pop()
        self.append(item)
        
        return item

    def sample_buffer_and_prepare_dataset(self, num):
        x_buffer, y_buffer = [], []
        for x, y in self.sample_buffer(num):
            x_buffer.append(x)
            y_buffer.append(y)

        x_buffer = np.array(x_buffer, dtype="float32")
        y_buffer = np.array(y_buffer, dtype="uint8")

        return x_buffer, y_buffer

    def sample_dataset_and_extend_buffer(self, dataset, num):
        items = self.sample(list(zip(*dataset)), num)
        self.extend(items)
    

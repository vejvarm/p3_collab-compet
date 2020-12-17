from collections import deque
import numpy as np
from utilities import transpose_list


class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)
        self.probs = deque(maxlen=self.size)

    def push(self, transition):
        """push into the buffer"""
        
        self.deque.append(transition[:-1])
        self.probs.append(transition[-1])

    def sample(self, batchsize, a=0.2):
        """sample from the buffer"""
        pa = np.power(self.probs, a).squeeze()
        p = pa/np.sum(pa)
        idcs = np.random.choice(np.arange(0, len(self.deque)), batchsize, p=p)
        samples = [self.deque[i] for i in idcs]

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)




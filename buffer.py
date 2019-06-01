
from dataclasses import dataclass, field
from typing import Any
import queue
import random
import numpy as np

@dataclass(order=True)
class Traject:
    priority: int
    samples: Any = field(compare=False)
    rewards: Any = field(compare=False)


class PriorityQueueSet(object):
    def __init__(self,length):
        self.que = queue.PriorityQueue(length)
        self.counter = 0
    def get_data(self):
        data = []
        while not(self.que.empty()):
            data.append(self.que.get())
        for i, d in enumerate(data):
            self.que.put(d)
        return data
    def sample(self):
        data = self.get_data()
        return random.sample(data,1)[0].samples
    def get_batches(self):
        data = self.get_data()
        result = []
        for d in data:
            result.append(d.samples)
        return result
    def add(self,samples, rewards):
        priority = rewards #np.mean(rewards)
        item = Traject(priority=priority,samples= samples, rewards = rewards)# actions=actions, observations = observations , values= values, rewards = rewards)
        if self.que.full():
            old_item = self.que.get()
            if item.priority > old_item.priority:
                self.que.put(item)
            else:
                self.que.put(old_item)
        else:
            self.counter+=1
            self.que.put(item)
    def mean_priority (self):
        return np.mean(np.asarray([i.priority for i in self.get_data()]))
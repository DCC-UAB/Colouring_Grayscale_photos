import random
from DataClass import *
from torchvision import transforms
import matplotlib.pyplot as plt

class LoaderClass():
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=1):
        self.dataset = dataset
        print(len(self.dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.batches_list = list()
        self.idx = 0

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.batches_list[idx]
    
    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration
        elem = self.dataset[self.idx]
        self.idx += 1
        return elem
    
    def __iter__(self):
        return iter(self.batches_list)

    def shuffle_data(self):
        self.dataset.shuffle()

    def batch_sampler(self):
        self.batches_list = list()
        num_batches, res = divmod(len(self.dataset), self.batch_size)
        for batch in range(num_batches):
            new_batch = list()
            for i in range(self.batch_size):
                new_batch.append(self.dataset[batch*self.batch_size + i])
            self.batches_list.append(new_batch)

        total_divided = num_batches*self.batch_size
        for i in range(res):
            idx = random.randint(0, num_batches-1)
            self.batches_list[idx].append(self.dataset[total_divided + i])
#作者：Yuwei
#链接：https://www.zhihu.com/question/307282137/answer/1560137140
#来源：知乎
from threading import Thread
from queue import Queue
import jax
import jax.numpy as jnp
import numpy as np

class CudaDataLoader:

    def __init__(self, loader, queue_size=2):
        self.loader = loader
        self.idx = 0
        self.queue = Queue(maxsize=queue_size)
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()
        self.val_train=0

    def load_loop(self):
        # The loop that will load into the queue in the background
        while True:
            for sample in self.loader:
                self.queue.put(jax.device_put(sample))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        elif self.idx > self.loader.train_length-0.5 and self.val_train < 0.5:
            self.val_train=1
            raise StopIteration
        elif self.idx > self.loader.length-0.5 and self.val_train > 0.5:
            self.val_train=0
            self.idx=0
            raise StopIteration
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out


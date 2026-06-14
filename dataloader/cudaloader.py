from queue import Queue
from threading import Thread

from src.jax_sharding import device_put_leading_axis_sharded, get_jax_devices, leading_axis_sharding


class CudaDataLoader:
    def __init__(self, loader, queue_size=2):
        self.idx = 0
        self.loader = loader
        self.queue = Queue(maxsize=queue_size)
        self.val_train = 0
        self.devices = get_jax_devices(loader.local_size)
        self.sharding = leading_axis_sharding(self.devices)
        self.worker = Thread(target=self.load_loop, daemon=True)
        self.worker.start()

    def load_loop(self):
        while True:
            for sample in self.loader:
                sample_set = (sample[0], self._transfer_to_gpus(sample[1:]))
                self.queue.put(sample_set)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.queue.join()
            self.worker.join()
            raise StopIteration
        elif self.idx > self.loader.ntrain - 0.5 and self.val_train < 0.5:
            self.val_train = 1
            raise StopIteration
        elif self.idx > self.loader.numpoint - 0.5 and self.val_train > 0.5:
            self.val_train = 0
            self.idx = 0
            raise StopIteration
        else:
            out = self.queue.get()
            self.idx = out[0]
            self.queue.task_done()
        return out[1]

    def _transfer_to_gpus(self, data):
        return device_put_leading_axis_sharded(data, self.sharding)

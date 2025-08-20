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
        self.devices = jax.local_devices()
        self.num_devices = len(self.devices)

    def load_loop(self):
        # The loop that will load into the queue in the background
        while True:
            for sample in self.loader:
                self.queue.put(self._transfer_to_gpus(sample))

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

    def _transfer_to_gpus(self, data):
        """
        使用JAX的惯用方法，将一个已经包含设备轴的Pytree分发到各设备。

        Args:
            data: 一个Pytree，其叶子节点的形状为 (num_devices, ...)。
                  例如: (a[2, 10, ...], (b[2, 10], c[2, 10]))

        Returns:
            一个单一的 ShardedDeviceArray 对象，其内部数据分布在所有设备上。
        """
        # 步骤1: 将一个 "Pytree of Arrays" 转置成一个 "List of Pytrees"。
        # 每个Pytree代表一个设备的数据。
        # jax.tree.map 会帮我们处理所有嵌套结构。
        list_of_shards = [
            jax.tree.map(lambda leaf: leaf[i], data)
            for i in range(self.num_devices)
        ]
        # 例如，list_of_shards[0] 会是：
        # (a[0], (b[0], c[0]))
        # 其中 a[0] 的形状是 (10, ...), b[0] 的形状是 (10,)

        # 步骤2: 使用 jax.device_put_sharded 创建一个 ShardedDeviceArray。
        # 这个函数是关键，它会保留数据的全局形状信息。
        return jax.device_put_sharded(list_of_shards, self.devices)

import random
from tinygrad.tensor import Tensor
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


class DataHandler:
    def __init__(
        self,
        data,
        test_ratio=0.2,
        seed=42,
        shuffle=True,
        transform_fn=None,
        num_workers=None,
        pool_type="process",
    ):
        """
        data: list of tuples
        test_ratio: fraction of data to use for testing
        seed: random seed for reproducibility
        shuffle: whether to shuffle data at the start
        transform_fn: function that processes a single sample
        num_workers: number of parallel workers (None or int)
        pool_type: "process" or "thread"

        •	If your transform_fn is CPU-bound (e.g., heavy computation
            like image augmentation), ProcessPoolExecutor is generally
            better since it can bypass the GIL.
        •	If your transform_fn is mostly I/O-bound (e.g., loading and
            decoding images from disk), ThreadPoolExecutor is often
            sufficient and can have lower overhead.

        If you don’t know which scenario dominates, ProcessPoolExecutor
        is usually a safer bet for heavy preprocessing tasks, as it
        scales better across multiple cores. The downside is that
        inter-process communication might have a bit more overhead
        than threads.

        """
        if transform_fn is None:
            raise ValueError("You must provide a transform_fn.")

        self.transform_fn = transform_fn
        self.shuffle = shuffle

        # Setup Executor if num_workers is specified
        self.executor = None
        if num_workers is not None and num_workers > 0:
            if pool_type == "thread":
                self.executor = ThreadPoolExecutor(max_workers=num_workers)
            else:
                # Default to process pool
                self.executor = ProcessPoolExecutor(max_workers=num_workers)

        # Split data
        self.train_data, self.test_data = self.train_test_split(data, test_ratio, seed)

        # Create index handlers for train and test splits
        self.train_samples = self.Samples(len(self.train_data), shuffle=shuffle)
        self.test_samples = self.Samples(len(self.test_data), shuffle=False)

    @staticmethod
    def train_test_split(data, test_ratio=0.2, seed=42):
        random.seed(seed)
        data_shuffled = data[:]
        random.shuffle(data_shuffled)
        test_size = int(len(data_shuffled) * test_ratio)
        test_data = data_shuffled[:test_size]
        train_data = data_shuffled[test_size:]
        return train_data, test_data

    class Samples:
        def __init__(self, length, shuffle=True):
            self.length = length
            self.shuffle = shuffle
            self.sample_idxs = list(range(length))
            if self.shuffle:
                random.shuffle(self.sample_idxs)

        def idxs(self, batch_size):
            assert batch_size <= self.length
            if len(self.sample_idxs) < batch_size:
                self.sample_idxs = list(range(self.length))
                if self.shuffle:
                    random.shuffle(self.sample_idxs)
            ret = self.sample_idxs[:batch_size]
            self.sample_idxs = self.sample_idxs[batch_size:]
            return ret

        def reset(self):
            self.sample_idxs = list(range(self.length))
            if self.shuffle:
                random.shuffle(self.sample_idxs)

    def make_batch(self, data, idxs):
        # Parallel or serial processing of samples
        if self.executor:
            processed = list(
                self.executor.map(self.transform_fn, [data[i] for i in idxs])
            )
        else:
            processed = [self.transform_fn(data[i]) for i in idxs]

        # Transpose the list of tuples
        transposed = list(zip(*processed))

        out = []
        for elements in transposed:
            if isinstance(elements[0], Tensor):
                out.append(Tensor.stack(*elements))
            else:
                out.append(elements)

        return tuple(out)

    def get_train_batch(self, batch_size):
        idxs = self.train_samples.idxs(batch_size)
        return self.make_batch(self.train_data, idxs)

    def get_test_batch(self, batch_size):
        idxs = self.test_samples.idxs(batch_size)
        return self.make_batch(self.test_data, idxs)

    def reset_train(self):
        self.train_samples.reset()

    def reset_test(self):
        self.test_samples.reset()

    def close(self):
        # Manually shut down the executor if it exists
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None

    def __del__(self):
        # Ensure executor is shut down on object destruction
        self.close()

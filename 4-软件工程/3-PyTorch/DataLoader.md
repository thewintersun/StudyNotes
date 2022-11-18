**[BUG] Pytorch nn.BatchNorm1d fails with batch size 1**

That's because you ==didn't use **drop_last=True** for your training dataloader== (now default in fastai) and ran in a batch of size 1, which causes error with BatchNorm during training (that's not us, it's on pytorch). You should use that option (or update your library).

**DataLoader**

https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html

```Python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)
```

Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.

The `DataLoader` supports both map-style and iterable-style datasets with single- or multi-process loading, customizing loading order and optional automatic batching (collation) and memory pinning.

See `torch.utils.data` documentation page for more details.

- Parameters

  **dataset** (*Dataset*) – dataset from which to load the data.

  **batch_size** (*int,* *optional*) – how many samples per batch to load (default: `1`).

  **shuffle** (*bool,* *optional*) – set to `True` to have the data reshuffled at every epoch (default: `False`).

  **sampler** (*Sampler,* *optional*) – defines the strategy to draw samples from the dataset. If specified, `shuffle`must be `False`.

  **batch_sampler** (*Sampler,* *optional*) – like `sampler`, but returns a batch of indices at a time. Mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`.

  **num_workers** (*int,* *optional*) – how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process. (default: `0`)

  **collate_fn** (*callable,* *optional*) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.

  **pin_memory** (*bool,* *optional*) – If `True`, the data loader will copy Tensors into CUDA pinned memory before returning them. If your data elements are a custom type, or your `collate_fn` returns a batch that is a custom type, see the example below.

  **drop_last** (*bool,* *optional*) – set to `True` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If `False` and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: `False`)

  **timeout** (*numeric,* *optional*) – if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: `0`)

  **worker_init_fn** (*callable,* *optional*) – If not `None`, this will be called on each worker subprocess with the worker id (an int in `[0, num_workers - 1]`) as input, after seeding and before data loading. (default: `None`)
import torch, os, errno
import torch.distributed as dist
from pathlib import Path
import torch
from collections import defaultdict

def atomic_torch_save(obj, f: str | Path, timer=None, **kwargs):
    f = str(f)
    temp_f = f + ".temp"
    torch.save(obj, temp_f, **kwargs)
    if timer is not None:
        timer.report(f'saving temp checkpoint')
    os.replace(temp_f, f)
    if timer is not None:
        timer.report(f'replacing temp checkpoint with checkpoint')
        return timer
    else:
        return

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    return

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

import os, time
from datetime import datetime

class Timer:
    '''
    This Timer can be integrated within a training routine to provide point-to-point
    script timing and reporting.

    def main():
        timer = Timer()
        time.sleep(2)
        timer.report("sleeping for 2 seconds")
        time.sleep(3)
        timer.report("sleeping for 3 seconds")

    >>> main()
    Start                                       0.000 ms     0.000 s total
    Completed sleeping for 2 seconds        2,000.000 ms     2.000 s total
    Completed sleeping for 3 seconds        3,000.000 ms     5.000 s total
    '''
    def __init__(self, report=None, start_time=None, running=0):
        self.start_time = start_time if start_time is not None else time.time()
        self.running = running
        if str(os.environ["RANK"]) == "0":
            report = report if report else "Start"
            print("[{:<80}] {:>12} ms, {:>12} s total".format(report, f'{0.0:,.3f}', f'{0.0:,.2f}'))
    def report(self, annot):
        if str(os.environ["RANK"]) == "0":
            now = time.time()
            duration = now - self.start_time
            self.running += duration
            print("Completed {:<70}{:>12} ms, {:>12} s total".format(annot, f'{1000*duration:,.3f}', f'{self.running:,.2f}'))
            self.start_time = now

class TimestampedTimer:
    '''
    This TimestampedTimer can be integrated within a training routine to provide 
    point-to-point script timing and reporting.

    def main():
        timer = TimestampedTimer()
        time.sleep(2)
        timer.report("sleeping for 2 seconds")
        time.sleep(3)
        timer.report("sleeping for 3 seconds")

    >>> main()
    [TIME] Start                                       0.000 ms     0.000 s total
    [TIME] Completed sleeping for 2 seconds        2,000.000 ms     2.000 s total
    [TIME] Completed sleeping for 3 seconds        3,000.000 ms     5.000 s total
    '''
    def __init__(self, report=None, start_time=None, running=0):
        if str(os.environ.get("RANK","NONE")) in ["0", "NONE"]:
            self.start_time = start_time if start_time is not None else time.time()
            self.running = running
            report = report if report else "Start"
            print("[ {} ] Completed {:<70}{:>12} ms, {:>12} s total".format(time.strftime("%Y-%m-%d %H:%M:%S"), report, f'{0.0:,.3f}', f'{0.0:,.2f}'))
    def report(self, annot):
        if str(os.environ.get("RANK","NONE")) in ["0", "NONE"]:
            now = time.time()
            duration = now - self.start_time
            self.running += duration
            print("[ {} ] Completed {:<70}{:>12} ms, {:>12} s total".format(time.strftime("%Y-%m-%d %H:%M:%S"), annot, f'{1000*duration:,.3f}', f'{self.running:,.2f}'))
            self.start_time = now

import math
import torch
from torch.utils.data import Dataset, DistributedSampler
from contextlib import contextmanager
from collections import defaultdict
from itertools import chain, repeat

class HasNotResetProgressError(Exception):
    pass

class AdvancedTooFarError(Exception):
    pass

class InterruptableDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        This is a DistributedSampler that can be suspended and resumed.

        This works by keeping track of the epoch and progress within the epoch.
        The progress is the number of samples that have been returned by the
        sampler. The epoch is the number of times the sampler has been iterated
        over.

        The epoch is incremented at the start of each epoch. The epoch is set
        to 0 at initialization.

        The progress is incremented by the number of samples returned by the
        sampler. The progress is reset to 0 at the end of each epoch.

        Suspending and resuming the sampler is done by saving and loading the
        state dict. The state dict contains the epoch and progress. This works
        because the permutation of the dataset is deterministic given the seed
        and epoch.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.progress = 0
        self._has_reset_progress = True

    def _reset_progress(self):
        self.progress = 0
        self._has_reset_progress = True

    def set_epoch(self, epoch):
        if not self._has_reset_progress:
            raise HasNotResetProgressError("You must reset progress before setting epoch e.g. `sampler.reset_progress()`\nor use `with sampler.in_epoch(epoch)` instead of `sampler.set_epoch(epoch)`")
        self.epoch = epoch

    def state_dict(self):
        return {"progress": self.progress, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.progress = state_dict["progress"]
        if not self.progress <= self.num_samples:
            raise AdvancedTooFarError(f"progress should be less than or equal to the number of samples. progress: {self.progress}, num_samples: {self.num_samples}")
        self.epoch = state_dict["epoch"]

    def advance(self, n: int):
        """
        Record that n samples have been consumed.
        """
        self.progress += n
        if self.progress > self.num_samples:
            raise AdvancedTooFarError("You have advanced too far. You can only advance up to the total size of the dataset.")

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # slice from progress to pick up where we left off
    
        for idx in indices[self.progress :]:
            yield idx

    @contextmanager
    def in_epoch(self, epoch):
        """
        This context manager is used to set the epoch. It is used like this:

        ```
        for epoch in range(0, 10):
            with sampler.in_epoch(epoch):
                for step, (x, ) in enumerate(dataloader):
                    # work would be done here...
        ```
        """
        self.set_epoch(epoch)
        yield
        self._reset_progress()

def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)
# --- MetricsTracker --- 

class MetricsTracker:
    '''
    This is a general purpose MetricsTracker to assist with recording metrics from
    a disributed cluster.

    The MetricsTracker is initialised without any prior knowledge of the metrics
    to be tracked.

    >>> metrics = MetricsTracker()

    Metrics can be accumulated as required, for example after each batch is procesed
    by the model, by passing a dictionary with metrics to be updated, then reduced 
    accross all nodes. Metric values are stored in a defaultdict.

    >>> preds = model(input)
    >>> loss = loss_fn(preds, targs)
    >>> metrics.update({"images_seen": len(images), "loss": loss.item()})
    >>> metrics.reduce()

    Metrics are assumed to be summable scalar values. After calling reduce(), the 
    metrics.local object contains the sum of corresponding metrics from all nodes
    which can be used for intermediate reporting or logging.

    >>> writer = SummaryWriter()
    >>> for metric,val in metrics.local.items():
    >>>     writer.add_scalar(metric, val, step)
    >>> writer.flush()
    >>> writer.close()

    Once all processing of the current batch has been completed, the MetricsTracker
    can be prepared for the next batch using reset_local().

    >>> metrics.reset_loca()

    Metrics are also accumulated for consecutive batches in the metrics.agg object.
    At the end of an epoch the MetricsTracker can be reset using end_epoch().

    >>> metrics.end_epoch()

    The MetricsTracker saves a copy of the accumulated metrics (metrics.agg) for
    each epoch which can be stored within a checkpoint.
    '''
    def __init__(self):
        self.local = defaultdict(float)
        self.agg = defaultdict(float)
        self.epoch_reports = []

    def update(self, metrics: dict):
        for m,v in metrics.items():
            self.local[m] += v
        
    def reduce(self):
        names, local = zip(*self.local.items())
        local = torch.tensor(local, dtype=torch.float16, requires_grad=False, device='cuda')
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        self.local = defaultdict(float, zip(names, local.cpu().numpy()))
        for k in self.local:
            self.agg[k] += self.local[k]

    def reset_local(self):
        self.local = defaultdict(float)
    
    def end_epoch(self):
        self.epoch_reports.append(dict(self.agg))
        self.local = defaultdict(float)
        self.agg = defaultdict(float)
    





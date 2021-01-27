from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# import distributed library of PyTorch
import torch.utils.data.distributed as torchDis

import horovod.torch as hvd

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            # add distributed training dataset sampler according to 
            # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
            self.train_sampler = torchDis.DistributedSampler(
                # taking concatenated datasets
                MyConcatDataset(datasets),

                # number of processes participarting in distributed training
                num_replicas = hvd.size(),

                # Rank of the current process within num_replicas
                rank = hvd.rank()
            )

            self.loader_train = dataloader.DataLoader(

                # taking concatenated datasets
                MyConcatDataset(datasets),

                # how many samples per batch to load
                batch_size=args.batch_size,

                # data shuffling disabled when using sampler
                shuffle=False,

                
                pin_memory=not args.cpu,

                # threads used in GPU for data loading
                num_workers=args.n_threads,

                # added the new distributed sampler
                sampler = self.train_sampler
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )

import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

# import horovod.torch
import horovod.torch as hvd

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    hvd.init()

    # cuda device flag
    args.cuda = args.n_GPUs > 1 and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            # added support for distributed training dataset loading
            # param added: hvd
            loader = data.Data(args,hvd)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None

            # wrapping optimizer with horovod distributed support
            # param added: hvd
            t = Trainer(args, loader, _model, _loss, checkpoint, hvd)
            while not t.terminate():
                hvd.broadcast_parameters(_model.state_dict(), root_rank = 0)
                hvd.broadcast_optimizer_state(t.optimizer, root_rank = 0)
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()

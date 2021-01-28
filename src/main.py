import torch
import torch.backends.cudnn as cudnn
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
    
    timer_HvdInit = utility.timer()
    # initialize horovod.torch
    hvd.init()

    # cuda device flag
    args.cuda = args.hvd and torch.cuda.is_available()
    print("args.cuda: "+str(args.cuda))
    print("args.hvd: "+str(args.hvd))
    print("cuda available: "+str(torch.cuda.is_available()))    
    # pinging local GPU to process
    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        print("hvd local rank:" + str(hvd.local_rank()))
        torch.cuda.manual_seed(args.seed)
    print("Horovod init time elapsed: "+str(timer_HvdInit.toc()))
    cudnn.benchmark = True

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
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None

            # wrapping optimizer with horovod distributed support
            # param added: hvd
            t = Trainer(args, loader, _model, _loss, checkpoint)
            timer_HvdBcast1 = utility.timer()
            hvd.broadcast_parameters(_model.state_dict(), root_rank = 0)
            print("Hvd Bcast params time elapsed: "+str(timer_HvdBcast1.toc()))
            timer_HvdBcast2 = utility.timer()
            hvd.broadcast_optimizer_state(t.optimizer, root_rank = 0)
            print("Hvd Bcast optimizer state time elapsed: "+str(timer_HvdBcast2.toc()))
            # Broadcast the initial variable states from rank 0 to all other processes
            while not t.terminate():
                timer_TrainLoop = utility.timer()
                t.train()
                print("Single loop time elapsed: "+str(timer_TrainLoop.toc()))

            checkpoint.done()

if __name__ == '__main__':
    main()

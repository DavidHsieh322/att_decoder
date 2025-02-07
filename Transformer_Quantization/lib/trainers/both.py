from ..tfqcores import TorchTrainer
from torch import optim
from torch import nn

class TFDecTrainerWithAdamWMSE(TorchTrainer):
    """
    A pytorch TFQ trainer with AdamW optimizer and MSELoss
    (decoder: Transformer Decoder for vector and code)
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFDecTrainerWithAdamWMSE, self).__init__(args, model)
        if args.encode_type == 'PQ':
            if args.sum_topk:
                self.model = model(args, args.source_dim // args.codebooks)
            else:
                self.model = model(args, args.source_dim * args.topk // args.codebooks)
        elif args.encode_type == 'RVQ':
            if args.sum_topk:
                self.model = model(args, args.source_dim)
            else:
                self.model = model(args, args.source_dim * args.topk)
        else:
            raise KeyError(f"Not support encode method: {args.encode_type}!")

        self.model = self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        self.loss_fn = nn.MSELoss()

        lr_milestones = [int(args.epochs*0.7), int(args.epochs*0.85)]
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=0.1
        )

    def _modelIO(self, args, xc, xv):
        """
        Inputs:
            args: parsed argument from config.py
            xc: codes after quantization
            xv: vectors from lookup table
        """
        xc, xv = xc.to(self.device), xv.to(self.device)

        bs = xv.shape[0]
        if args.sum_topk:
            xv = xv.sum(axis=2)
        else:
            xv = xv.reshape(bs, args.codebooks, -1)

        pred = self.model(xc, xv)

        return pred
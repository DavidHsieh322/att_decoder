from ..tfqcores import TorchPredictor
import torch

class TFDecPredictor(TorchPredictor):
    """
    A pytorch TFQ predictor
    (decoder: Transformer Decoder for vector and code)
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFDecPredictor, self).__init__(args, model)
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

        model_weights = torch.load(self.load_path)
        self.model.load_state_dict(model_weights)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
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
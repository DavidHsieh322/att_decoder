from .both import TFDecPredictor
import torch

class TFEncPredictor(TFDecPredictor):
    """
    A pytorch TFQ Predictor
    (decoder: Transformer encoder for vectors)
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFEncPredictor, self).__init__(args, model)

    @torch.no_grad()
    def _modelIO(self, args, xc, xv):
        """
        Inputs:
            args: parsed argument from config.py
            xc: codes after quantization
            xv: vectors from lookup table
        """
        xv = xv.to(self.device)

        bs = xv.shape[0]
        if args.sum_topk:
            xv = xv.sum(axis=2)
        else:
            xv = xv.reshape(bs, args.codebooks, -1)

        pred = self.model(xv)

        return pred

class TFDecTopKPredictor(TFDecPredictor):
    """
    A pytorch TFQ predictor
    (decoder: Transformer decoder for only vectors)
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFDecTopKPredictor, self).__init__(args, model)

    @torch.no_grad()
    def _modelIO(self, args, xc, xv):
        """
        Inputs:
            args: parsed argument from config.py
            xc: codes after quantization
            xv: vectors from lookup table
        """
        xv = xv.to(self.device)

        bs = xv.shape[0]
        if args.sum_topk:
            xqv = xv.sum(axis=2)
        else:
            xqv = xv.reshape(bs, args.codebooks, -1)

        xv = xv.reshape(bs, args.codebooks * args.topk, -1)

        pred = self.model(xqv, xv)

        return pred

class TFDecTop1QPredictor(TFDecPredictor):
    """
    A pytorch TFQ predictor
    (decoder: Transformer decoder for only vectors)
    (query contains only top1 vector)
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFDecTop1QPredictor, self).__init__(args, model)

    @torch.no_grad()
    def _modelIO(self, args, xc, xv):
        """
        Inputs:
            args: parsed argument from config.py
            xc: codes after quantization
            xv: vectors from lookup table
        """
        xv = xv.to(self.device)
        bs = xv.shape[0]
        xvTop1 = xv[:, :, 0, :]
        xv = xv.reshape(bs, args.codebooks * args.topk, -1)
        pred = self.model(xvTop1, xv)

        return pred
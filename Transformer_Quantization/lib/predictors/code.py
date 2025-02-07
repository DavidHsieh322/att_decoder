from .both import TFDecPredictor
import torch

class TFEncPredictor(TFDecPredictor):
    """
    A pytorch TFQ predictor
    (decoder: Transformer encoder for codes)
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
        xc = xc.to(self.device)
        pred = self.model(xc)

        return pred

class TFDecTopKPredictor(TFDecPredictor):
    """
    A pytorch TFQ predictor
    (decoder: Transformer decoder for only codes)
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
        xc = xc.to(self.device)
        pred = self.model(xc, xc)

        return pred

class TFDecTop1QPredictor(TFDecPredictor):
    """
    A pytorch TFQ predictor
    (decoder: Transformer decoder for only codes)
    (query contains only top1 code)
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
        xc = xc.to(self.device)
        bs = xc.shape[0]
        xcTop1 = xc.view(bs, args.codebooks, -1)[:, :, 0]
        pred = self.model(xcTop1, xc)

        return pred
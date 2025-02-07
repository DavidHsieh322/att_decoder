from .both import TFDecTrainerWithAdamWMSE
from ..tfqcores import TorchTrainer

class TFEncTrainer(TorchTrainer):
    """
    A pytorch TFQ trainer prototype with Transformer encoder for vectors
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFEncTrainer, self).__init__(args, model)

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

class TFDecTopKTrainer(TorchTrainer):
    """
    A pytorch TFQ trainer prototype with Transformer decoder for only vectors
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFDecTopKTrainer, self).__init__(args, model)

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

class TFDecTop1QTrainer(TorchTrainer):
    """
    A pytorch TFQ trainer prototype with Transformer decoder for only vectors
    (query contains only top1 vector)
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFDecTop1QTrainer, self).__init__(args, model)

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

class TFEncTrainerWithAdamWMSE(TFEncTrainer, TFDecTrainerWithAdamWMSE):
    """
    A pytorch TFQ trainer with AdamW optimizer and MSELoss
    (decoder: Transformer encoder for vectors)
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFEncTrainerWithAdamWMSE, self).__init__(args, model)

class TFDecTopKTrainerWithAdamWMSE(TFDecTopKTrainer, TFDecTrainerWithAdamWMSE):
    """
    A pytorch TFQ trainer with AdamW optimizer and MSELoss
    (decoder: Transformer decoder for only vectors)
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFDecTopKTrainerWithAdamWMSE, self).__init__(args, model)

class TFDecTop1QTrainerWithAdamWMSE(TFDecTop1QTrainer, TFDecTrainerWithAdamWMSE):
    """
    A pytorch TFQ trainer with AdamW optimizer and MSELoss
    (decoder: Transformer decoder for only vectors)
    (query contains only top1 vector)
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFDecTop1QTrainerWithAdamWMSE, self).__init__(args, model)
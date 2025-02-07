from .both import TFDecTrainerWithAdamWMSE
from ..tfqcores import TorchTrainer

class TFEncTrainer(TorchTrainer):
    """
    A pytorch TFQ trainer prototype with Transformer encoder for codes
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
        xc = xc.to(self.device)
        pred = self.model(xc)

        return pred

class TFDecTopKTrainer(TorchTrainer):
    """
    A pytorch TFQ trainer prototype with Transformer decoder for only codes
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
        xc = xc.to(self.device)
        pred = self.model(xc, xc)

        return pred

class TFDecTop1QTrainer(TorchTrainer):
    """
    A pytorch TFQ trainer prototype with Transformer decoder for only codes
    (query contains only top1 code)
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
        xc = xc.to(self.device)
        bs = xc.shape[0]
        xcTop1 = xc.view(bs, args.codebooks, -1)[:, :, 0]
        pred = self.model(xcTop1, xc)

        return pred

class TFEncTrainerWithAdamWMSE(TFEncTrainer, TFDecTrainerWithAdamWMSE):
    """
    A pytorch TFQ trainer with AdamW optimizer and MSELoss
    (decoder: Transformer encoder for codes)
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
    (decoder: Transformer decoder for only codes)
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
    (decoder: Transformer decoder for only codes)
    (query contains only top1 code)
    """
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        super(TFDecTop1QTrainerWithAdamWMSE, self).__init__(args, model)
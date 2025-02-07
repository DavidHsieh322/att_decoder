from .tools import Timer
from .tools import Plotter
from .metrics import compute_reconstruction_error
from torch.utils.data import DataLoader, random_split
import torch
import os
import pandas as pd

class TorchTrainer():
    """A pytorch trainer prototype for Transformer Quantization"""
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        print(f"Using {self.device} device")
        print('='*40)

        if args.sum_topk:
            topk_type = 'sum'
        else:
            topk_type = 'concat'

        self.exp_name = f"{args.exp_name}-{args.encode_type}"
        self.exp_name += f"-{args.model}-{args.codebooks}m-top{args.topk}-{topk_type}"
        self.timer = Timer(args.epochs)
        self.plotter = Plotter()
        self.stop_count = 0
        self.stop_flag = False
        self.best_loss = None
        self.best_weights = None
        self.history = {}
        self.history['epoch'] = []
        self.history['train_loss'] = []
        self.history['val_loss'] = []

        # self.model = None
        # self.optimizer = None
        # self.loss_fn = None
        self.scheduler = None

    def _build_dataloader(self, args, data):
        valNum = int(len(data) * 0.1)
        trainNum = len(data) - valNum
        train_data, val_data = random_split(
            data, [trainNum, valNum], 
            generator=torch.Generator().manual_seed(1)
        )

        persistent = (args.workers != 0)
        self.train_dataloader = DataLoader(
            train_data, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.workers, 
            persistent_workers=persistent
        )
        self.val_dataloader = DataLoader(
            val_data, 
            batch_size=args.batch_size, 
            num_workers=args.workers, 
            persistent_workers=persistent
        )

    def _modelIO(self, args, xc, xv):
        """
        Inputs:
            args: parsed argument from config.py
            xc: codes after quantization
            xv: vectors from lookup table
        """
        xc, xv = xc.to(self.device), xv.to(self.device)
        pred = self.model(xc, xv)

        return pred

    def _train_loop(self, args) -> float:
        """
        Inputs:
            args: parsed argument from config.py
        """
        size = len(self.train_dataloader.dataset)
        current = 0
        train_loss = 0
        self.model.train()

        for batch, (Xc, Xv, y) in enumerate(self.train_dataloader):
            y = y.to(self.device)

            # Compute prediction and loss
            pred = self._modelIO(args, Xc, Xv)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=args.max_norm)
            self.optimizer.step()

            loss = loss.item()
            train_loss += loss

            current += len(Xc)
            if batch % 5 == 0:
                print(f"loss: {loss:7f}  [{current:5d}/{size:5d}]", end = '\r')
        print()
    
        train_loss /= (batch + 1)
        print(f"Train Error: \n Avg loss: {train_loss:8f} \n")

        return train_loss

    def _val_loop(self, args) -> float:
        """
        Inputs:
            args: parsed argument from config.py
        """
        size = len(self.val_dataloader.dataset)
        current = 0
        val_loss = 0
        self.model.eval()

        with torch.no_grad():
            for batch, (Xc, Xv, y) in enumerate(self.val_dataloader):
                y = y.to(self.device)

                pred = self._modelIO(args, Xc, Xv)
                loss = self.loss_fn(pred, y).item()
                val_loss += loss

                current += len(Xc)
                if batch % 5 == 0:
                    print(f"loss: {loss:7f}  [{current:5d}/{size:5d}]", end = '\r')
            print()

        val_loss /= (batch + 1)
        print(f"Val Error: \n Avg loss: {val_loss:8f} \n")

        return val_loss

    def _early_stop_check(self, args, val_loss: float) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            val_loss: average validation loss of each epoch
        """
        if self.best_loss and self.best_loss < val_loss:
            self.stop_count += 1

        else:
            self.best_weights = self.model.state_dict()
            self.best_loss = val_loss
            self.stop_count = 0

        if self.stop_count == args.early_stop:
            self.model.load_state_dict(self.best_weights)
            self.stop_flag = True

    def _save_checkpoint(self, args, exp_No: int, checkpoint: int) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            checkpoint: epoch
        """
        # save model weights
        save_path = f"{args.results_dir}/{self.exp_name}-{exp_No}/"
        save_path += f"weights/{self.exp_name}-{exp_No}_{checkpoint}.pth"
        torch.save(self.model.state_dict(), save_path)

        # save loss log
        log_df = pd.DataFrame.from_dict(self.history)
        log_df = pd.concat([log_df])
        log_df.set_index('epoch', inplace=True)
        log_df.to_csv(f"{args.results_dir}/{self.exp_name}-{exp_No}/log.csv")

        # save loss curve
        savefig_path = (f"{args.results_dir}/{self.exp_name}-{exp_No}/loss.png")
        self.plotter.reset_dataList()
        self.plotter.append_data(self.history['train_loss'], 'b-', 'training')
        if self.history['val_loss']:
            self.plotter.append_data(self.history['val_loss'], 'r-', 'validation')
        self.plotter.line_chart(
            savefig_path, 
            'Epochs', 
            'Loss', 
            1, 
            checkpoint, 
            args.plotter_x_interval
        )

    def fit(self, args, data) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            data: a torch dataset object (Dataset class instance)
        """
        if not os.path.isdir(args.results_dir):
            os.mkdir(args.results_dir)

        if args.exp_No:
            exp_No = args.exp_No
            if not os.path.isdir(f"{args.results_dir}/{self.exp_name}-{exp_No}"):
                os.mkdir(f"{args.results_dir}/{self.exp_name}-{exp_No}")
        else:
            exp_No = 1
            while os.path.isdir(f"{args.results_dir}/{self.exp_name}-{exp_No}"):
                exp_No += 1
            os.mkdir(f"{args.results_dir}/{self.exp_name}-{exp_No}")

        if not os.path.isdir(f"{args.results_dir}/{self.exp_name}-{exp_No}/weights"):
            os.mkdir(f"{args.results_dir}/{self.exp_name}-{exp_No}/weights")

        args_txt = ''
        for k, v in vars(args).items():
            args_txt += f"{k} = {v}\n"
        with open(f"{args.results_dir}/{self.exp_name}-{exp_No}/args.txt", 'w') as f:
            f.write(args_txt)

        self._build_dataloader(args, data)

        for epoch in range(args.epochs):
            self.timer.start()

            print(f"Epoch {epoch + 1}")
            print('-'*40)
            self.history['epoch'].append(epoch + 1)

            train_loss = self._train_loop(args)
            self.history['train_loss'].append(train_loss)

            if self.scheduler:
                self.scheduler.step()

            if self.val_dataloader:
                val_loss = self._val_loop(args)
                self.history['val_loss'].append(val_loss)
                self._early_stop_check(args, val_loss)

            if (epoch + 1) % args.checkpoint == 0:
                self._save_checkpoint(args, exp_No, epoch + 1)
            
            if self.stop_flag:
                break

            self.timer.finish()

            print(f"Time: {self.timer.elapsed_time:.2f} sec, ETA: {self.timer.ETA}\n")

        if (epoch + 1) % args.checkpoint != 0:
            self._save_checkpoint(args, exp_No, epoch + 1)

class TorchPredictor():
    """Basic pytorch predictor for Transformer Quantization"""
    def __init__(self, args, model) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            model: a torch model class (inheriting from nn.Module, nn.Sequential, ...)
        """
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        print(f"Using {self.device} device")
        print('='*40)

        if args.sum_topk:
            topk_type = 'sum'
        else:
            topk_type = 'concat'

        self.exp_name = f"{args.exp_name}-{args.encode_type}"
        self.exp_name += f"-{args.model}-{args.codebooks}m-top{args.topk}-{topk_type}"
        self.timer = Timer(args.epochs)
        self.preds = torch.tensor([], device=self.device)
        self.gts = torch.tensor([])

        self.load_path = f"{args.results_dir}/{self.exp_name}-{args.exp_No}/"
        self.load_path += f"weights/{self.exp_name}-{args.exp_No}_{args.checkpoint}.pth"

        # self.model = None
        # model_weights = torch.load(self.load_path)
        # self.model.load_state_dict(model_weights)
        # self.model = self.model.to(self.device)
        # self.model.eval()

    def _build_dataloader(self, args, data):
        self.dataloader = DataLoader(
            data, 
            batch_size=args.batch_size, 
            num_workers=args.workers
        )

    @torch.no_grad()
    def _modelIO(self, args, xc, xv):
        """
        Inputs:
            args: parsed argument from config.py
            xc: codes after quantization
            xv: vectors from lookup table
        """
        xc, xv = xc.to(self.device), xv.to(self.device)
        pred = self.model(xc, xv)

        return pred

    @torch.no_grad()
    def _predict(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        size = len(self.dataloader.dataset)
        current = 0

        for batch, (Xc, Xv, y) in enumerate(self.dataloader):

            pred = self._modelIO(args, Xc, Xv)
            self.preds = torch.cat([self.preds, pred], dim=0)
            self.gts = torch.cat([self.gts, y], dim=0)

            current += len(Xc)
            if batch % 5 == 0:
                print(f"Progress: [{current:5d}/{size:5d}]", end = '\r')
        print()

    @torch.no_grad()
    def evaluate(self, args, data, origRecVecs) -> None:
        """
        Inputs:
            args: parsed argument from config.py
            data: a torch dataset object (Dataset class instance)
            origRecVecs: the vectors reconstructed in the original way
        """
        self._build_dataloader(args, data)

        self.timer.start()

        self._predict(args)
        self.preds = self.preds.cpu()

        self.timer.finish()
        print(f"Detection Time: {self.timer.elapsed_time:.2f} sec\n")

        origRecVecs = torch.as_tensor(origRecVecs).type(torch.float32)
        orig_recon_err = compute_reconstruction_error(origRecVecs, self.gts)
        attn_recon_err = compute_reconstruction_error(self.preds, self.gts)

        print('Reconstruction Error:')
        print(f"Original Error:      {orig_recon_err:.4f}")
        print(f"Attn. Decoder Error: {attn_recon_err:.4f}")
from ..metrics import compute_reconstruction_error
from ..tools import get_code_dtype
import faiss
import os
import pandas as pd
import numpy as np

class FaissPQPredictor():
    """Basic product quantization predictor"""
    def __init__(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        query_path = f"{args.data_dir}/{args.dataset}_Query.txt"
        self.codebooks_path = f"{args.cb_dir}/{args.dataset}_PQ_{args.codebooks}x{args.cb_bits}.npy"
        self.codes_path = f"{args.code_dir}/{args.dataset}_PQ_{args.codebooks}x{args.cb_bits}_query.npy"

        self.query = pd.read_csv(query_path, sep='\t', header=None)
        self.query = self.query.drop(columns=[args.source_dim])

        self.codebooks = None
        self.code_dtype = get_code_dtype(args.cb_bits)

    def _preprocess(self, args, df: pd.DataFrame) -> np.ndarray:
        """
        Inputs:
            args: parsed argument from config.py
            df: data before preprocessing
        """
        out = df.to_numpy().astype(np.float32) / 255.0
        out = out.reshape(out.shape[0], args.codebooks, -1)
        out = out.transpose(1, 0, 2)
        out = np.ascontiguousarray(out)

        return out

    def _auto_encode(self, args, query: np.ndarray) -> np.ndarray:
        """
        Inputs:
            args: parsed argument from config.py
            query: test data after preprocessing
        """
        self.codes = np.array([], dtype=self.code_dtype)
        self.codes = self.codes.reshape(len(self.query), 0)

        query_r = np.array([], dtype=np.float32).reshape(query.shape[1], 0)

        for x, cb in zip(query, self.codebooks):
            index = faiss.IndexFlatL2(args.source_dim // args.codebooks)
            index.add(cb)
            D, I = index.search(x, 1)
            x = cb[I].squeeze(1)
            self.codes = np.concatenate([self.codes, I.astype(self.code_dtype)], axis=1)
            query_r = np.concatenate([query_r, x], axis=1)

        return query_r

    def save_codes(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        if not os.path.isdir(args.code_dir):
            os.mkdir(args.code_dir)

        with open(self.codes_path, 'wb') as f:
            np.save(f, self.codes)

    def predict(self, args) -> np.ndarray:
        """
        Inputs:
            args: parsed argument from config.py
        """
        if not self.codebooks:
            with open(self.codebooks_path, 'rb') as f:
                self.codebooks = np.load(f)

        query = self._preprocess(args, self.query)
        query_r = self._auto_encode(args, query)

        return query_r

    def evaluate(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        if not self.codebooks:
            with open(self.codebooks_path, 'rb') as f:
                self.codebooks = np.load(f)

        query = self._preprocess(args, self.query)
        query_r = self._auto_encode(args, query)
        query = query.transpose(1, 0, 2).reshape(query.shape[1], -1)
        recon_err = compute_reconstruction_error(query, query_r)
        print('='*30)
        print(f"Reconstruction Error: {recon_err:.4f}")

class FaissRVQPredictor(FaissPQPredictor):
    """Basic residual vector quantization predictor"""
    def __init__(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        super(FaissRVQPredictor, self).__init__(args)
        self.codebooks_path = f"{args.cb_dir}/{args.dataset}_RVQ_{args.codebooks}x{args.cb_bits}.npy"
        self.codes_path = f"{args.code_dir}/{args.dataset}_RVQ_{args.codebooks}x{args.cb_bits}_query.npy"

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Inputs:
            df: data before preprocessing
        """
        out = df.to_numpy().astype(np.float32) / 255.0
        out = np.ascontiguousarray(out)

        return out

    def _auto_encode(self, args, query: np.ndarray) -> np.ndarray:
        """
        Inputs:
            args: parsed argument from config.py
            query: test data after preprocessing
        """
        self.codes = np.array([], dtype=self.code_dtype)
        self.codes = self.codes.reshape(len(self.query), 0)

        query_r = np.zeros_like(query)
        query_t = query

        for cb in self.codebooks:
            index = faiss.IndexFlatL2(args.source_dim)
            index.add(cb)
            D, I = index.search(query_t, 1)
            x = cb[I].squeeze(1)
            query_t = query_t - x
            query_r = query_r + x
            self.codes = np.concatenate([self.codes, I.astype(self.code_dtype)], axis=1)

        return query_r

    def evaluate(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        if not self.codebooks:
            with open(self.codebooks_path, 'rb') as f:
                self.codebooks = np.load(f)

        query = self._preprocess(self.query)
        query_r = self._auto_encode(args, query)
        recon_err = compute_reconstruction_error(query, query_r)
        print('='*30)
        print(f"Reconstruction Error: {recon_err:.4f}")
from ..tools import get_code_dtype
import faiss
import os
import pandas as pd
import numpy as np

class FaissPQTrainer():
    """Basic product quantization trainer"""
    def __init__(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        database_path = f"{args.data_dir}/{args.dataset}.txt"
        self.codebooks_path = f"{args.cb_dir}/{args.dataset}_PQ_{args.codebooks}x{args.cb_bits}.npy"
        self.codes_path = f"{args.code_dir}/{args.dataset}_PQ_{args.codebooks}x{args.cb_bits}.npy"

        self.database = pd.read_csv(database_path, sep='\t', header=None)
        self.database = self.database.drop(columns=[args.source_dim])

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

    def _save_codebooks(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        if not os.path.isdir(args.cb_dir):
            os.mkdir(args.cb_dir)

        with open(self.codebooks_path, 'wb') as f:
            np.save(f, self.codebooks)

    def _save_codes(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        if not os.path.isdir(args.code_dir):
            os.mkdir(args.code_dir)

        with open(self.codes_path, 'wb') as f:
            np.save(f, self.codes)
    
    def fit(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        self.codebooks = np.array([], dtype=np.float32)
        self.codebooks = self.codebooks.reshape(0, 2**args.cb_bits, args.source_dim // args.codebooks)

        self.codes = np.array([], dtype=self.code_dtype)
        self.codes = self.codes.reshape(len(self.database), 0)

        gpu = False if args.device == 'cpu' else faiss.get_num_gpus()
        for x in self._preprocess(args, self.database):
            kmeans = faiss.Kmeans(
                args.source_dim // args.codebooks, 
                2**args.cb_bits, 
                niter=args.kmeans_iter, 
                nredo=args.kmeans_redo, 
                verbose=True, 
                gpu=gpu
            )
            kmeans.cp.max_points_per_centroid = x.shape[0] // 2**args.cb_bits + 1
            kmeans.train(x)
            D, I = kmeans.index.search(x, 1)
            self.codes = np.concatenate([self.codes, I.astype(self.code_dtype)], axis=1)
            self.codebooks = np.concatenate([self.codebooks, np.expand_dims(kmeans.centroids, axis=0)])

        self._save_codes(args)
        self._save_codebooks(args)

class FaissRVQTrainer(FaissPQTrainer):
    """Basic residual vector quantization trainer"""
    def __init__(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        super(FaissRVQTrainer, self).__init__(args)
        self.codebooks_path = f"{args.cb_dir}/{args.dataset}_RVQ_{args.codebooks}x{args.cb_bits}.npy"
        self.codes_path = f"{args.code_dir}/{args.dataset}_RVQ_{args.codebooks}x{args.cb_bits}.npy"

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Inputs:
            args: parsed argument from config.py
            df: data before preprocessing
        """
        out = df.to_numpy().astype(np.float32) / 255.0
        out = np.ascontiguousarray(out)

        return out

    def fit(self, args) -> None:
        """
        Inputs:
            args: parsed argument from config.py
        """
        data = self._preprocess(self.database)

        self.codebooks = np.array([], dtype=np.float32)
        self.codebooks = self.codebooks.reshape(0, 2**args.cb_bits, args.source_dim)

        self.codes = np.array([], dtype=self.code_dtype)
        self.codes = self.codebooks.reshape(len(self.database), 0)

        gpu = False if args.device == 'cpu' else faiss.get_num_gpus()
        for i in range(args.codebooks):
            kmeans = faiss.Kmeans(
                args.source_dim, 
                2**args.cb_bits, 
                niter=args.kmeans_iter, 
                nredo=args.kmeans_redo, 
                verbose=True, 
                gpu=gpu
            )
            kmeans.cp.max_points_per_centroid = data.shape[0] // 2**args.cb_bits + 1
            kmeans.train(data)
            D, I = kmeans.index.search(data, 1)
            data = data - kmeans.centroids[I].squeeze(1)
            self.codes = np.concatenate([self.codes, I.astype(self.code_dtype)], axis=1)
            self.codebooks = np.concatenate([self.codebooks, np.expand_dims(kmeans.centroids, axis=0)])

        self._save_codes(args)
        self._save_codebooks(args)
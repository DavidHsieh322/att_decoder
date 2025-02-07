from torch.utils.data import Dataset
import torch
import faiss
import os
import pandas as pd
import numpy as np

class TopKCodes(Dataset):
    """Basic data reading pipeline"""
    def __init__(self, args, codes_path: str, data_path: str, codebooks_path: str) -> None:
        """
        Inputs:
            args: args: parsed argument from config.py
            codes_path: codes file path
            data_path: data file path
            codebooks_path: pretrained codebooks file path
        """
        self.data = pd.read_csv(data_path, sep='\t', header=None)
        self.data = self.data.drop(columns=[args.source_dim])

        with open(codebooks_path, 'rb') as f:
            self.codebooks = np.load(f)

        with open(codes_path, 'rb') as f:
            self.codes = np.load(f)

        self.indices = []
        for cb in self.codebooks:
            index = faiss.IndexFlatL2(self.codebooks.shape[-1])
            index.add(cb)
            self.indices.append(index)

        self.topk = args.topk

    def __len__(self) -> int:
        """
        Inputs:
            None
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Inputs:
            idx: data index number
        """
        source = self.data.iloc[idx].to_numpy()
        source = torch.as_tensor(source).type(torch.float32) / 255.0
        codes_t = self.codes[idx].astype(np.int64)

        codevecs = torch.tensor([], dtype=torch.float32)
        codes = torch.tensor([], dtype=torch.int64)

        for c, cb, index in zip(codes_t, self.codebooks, self.indices):
            x = np.expand_dims(cb[c], axis=0)
            c = index.search(x, self.topk)[1]
            x = torch.as_tensor(cb[c]).type(torch.float32)
            c = torch.as_tensor(c).view(-1).type(torch.int64)
            codes = torch.concat([codes, c], dim=0)
            codevecs = torch.concat([codevecs, x], dim=0)

        return codes, codevecs, source
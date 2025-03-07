import torch
from torch.utils.data import Dataset

from typing import Union
from pathlib import Path

from src.rinalmo.data.alphabet import Alphabet

import pickle as pkl
import numpy as np

class ContactDataset(Dataset):
    def __init__(
        self,
        datapath: Union[str, Path],
        alphabet: Alphabet = Alphabet(),
        min_seq_len: int = 0,
        max_seq_len: int = 999_999_999,
    ):
        super().__init__()

        self.alphabet = alphabet
        self.datapath = Path(datapath)

        with open(datapath, 'rb') as f:
            pairs = pkl.load(f)
        out = list(map(list, zip(*pairs)))
        self.seqs, arrs = out[0], out[1]
        self.tgts = arrs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):

        seq = self.seqs[idx]
        seq_encoded = torch.tensor(self.alphabet.encode(str(seq)), dtype=torch.int64)
        contact_tgt = torch.tensor(self.tgts[idx], dtype=torch.int64)

        return str(seq), seq_encoded, contact_tgt

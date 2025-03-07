from torch.utils.data import DataLoader

import lightning.pytorch as pl

from typing import Union, Optional
from pathlib import Path

from src.rinalmo.data.alphabet import Alphabet
from src.rinalmo.data.downstream.contact.dataset import ContactDataset

# Default train/val/test directory names
TRAIN_DIR_NAME = "train"
VAL_DIR_NAME = "valid"
TEST_DIR_NAME = "test"

class ContactDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Union[Path, str],
        alphabet: Alphabet = Alphabet(),
        num_workers: int = 0,
        pin_memory: bool = False,
        min_seq_len: int = 0,
        max_seq_len: int = 999_999_999,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.alphabet = alphabet

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        self.train_f = '3d-contact-set-train-c.pkl'
        self.val_f = '3d-contact-set-test-c.pkl'

    def setup(self, stage: Optional[str] = None):

        # TODO: Provide dataset clustered by homology instead of randomly splitting entire dataset
        self.stage = stage

        self.train_dataset = \
            ContactDataset(
                self.data_root / self.train_f,
                self.alphabet,
                min_seq_len=self.min_seq_len,
                max_seq_len=self.max_seq_len,
            )
        self.val_dataset = \
            ContactDataset(
                self.data_root / self.val_f,
                self.alphabet,
                min_seq_len=self.min_seq_len,
                max_seq_len=self.max_seq_len,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

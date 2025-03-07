from typing import List

from src.rinalmo.data.constants import *

class Alphabet:
    def __init__(
        self,
        standard_tkns: List[str] = RNA_TOKENS,
        special_tkns: List[str] = [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
    ):
        super().__init__()

        self.standard_tkns = standard_tkns
        self.special_tkns = special_tkns

        self.idx_to_tkn = special_tkns + standard_tkns
        self.tkn_to_idx = {t : i for i, t in enumerate(self.idx_to_tkn)}

        self.cls_idx = self.tkn_to_idx[CLS_TKN]
        self.eos_idx = self.tkn_to_idx[EOS_TKN]

        self.unk_idx = self.tkn_to_idx[UNK_TKN]
        self.pad_idx = self.tkn_to_idx[PAD_TKN]
        self.mask_idx = self.tkn_to_idx[MASK_TKN]

    def __len__(self):
        return len(self.idx_to_tkn)

    def get_tkn(self, idx):
        return self.idx_to_tkn[idx]

    def get_idx(self, tkn):
        return self.tkn_to_idx.get(tkn, self.unk_idx)

    def encode(self, seq: str, pad_to_len: int = -1):
        seq = seq.upper()
        # Replace uracil tokens with thymine tokens
        seq = seq.replace(THYMINE_TKN, URACIL_TKN)

        encoded_seq = [self.cls_idx] + [self.get_idx(tkn) for tkn in seq] + [self.eos_idx]
        if len(encoded_seq) < pad_to_len:
            encoded_seq = encoded_seq + (pad_to_len - len(encoded_seq)) * [self.pad_idx]

        return encoded_seq
    
    def batch_tokenize(self, seqs: List[str]):
        max_len = max(len(seq) for seq in seqs)
        max_len += 2 # CLS and EOS

        batch = []
        for seq in seqs:
            batch.append(self.encode(seq, pad_to_len=max_len))

        return batch

    def decode(self, xt):
        """
        Decodes a batch of tokenized sequences into strings using a tokenizer.

        Parameters:
            xt (torch.Tensor or np.ndarray): A batch of tokenized sequences with shape [B, L],
                                            where B is the batch size, and L is the sequence length.
            tokenizer (object): Tokenizer object with a `get_tkn(idx)` method to map indices to tokens.

        Returns:
            List[str]: A list of decoded strings, one for each sequence in the batch.
        """
        decoded_strings = []
        
        for sequence in xt:  # Iterate through each sequence in the batch
            decoded_sequence = ''.join([self.get_tkn(idx) for idx in sequence[1:-1]])  # Decode tokens and join
            decoded_strings.append(decoded_sequence)
        
        return decoded_strings


# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright (C) 2024 Atom Bioworks


from src.ncrna.models.lm.lm_mod import LM_pretrained
from dataclasses import dataclass, field
import torch
from Bio import SeqIO


@dataclass
class NetConfig:
    arch_type: str = "lm"
    name: str = "default"
    dropout: float = 0.1
    pretrain: bool = False
    pretrained_model_name_or_path: str = ""

@dataclass
class LoRAConfig:
    lora: bool = field(
        default=False
    )
    lora_rank: int = field(
        default=16
    )
    lora_dropout: float = field(
        default=0.1
    )
    lora_target_module: str = field(
        default=""
    )
    modules_to_save: str = field(
        default=""
    )
    
def get_net(cfg):
    if cfg.net.arch_type == 'lm':
        net = LM_pretrained()
    else:
        raise NotImplementedError
    
    return net

def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking

def read_fasta(file):
    """
    Reads a FASTA file and extracts the headers and sequences.
    
    Parameters:
    file (str): Path to the input FASTA file.
    
    Returns:
    tuple: A tuple containing two lists:
           - headers: A list of sequence headers (without the '>' symbol).
           - sequences: A list of sequences.
    """
    headers = []
    sequences = []
    
    # Parse the FASTA file
    for record in SeqIO.parse(file, "fasta"):
        headers.append(record.id)  # Add the header (without the '>' symbol)
        sequences.append(str(record.seq))  # Add the sequence as a string
    
    return headers, sequences

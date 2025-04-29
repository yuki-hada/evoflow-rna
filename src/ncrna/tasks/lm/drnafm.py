
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright (C) 2024 Atom Bioworks


from dataclasses import dataclass, field
import torch
import torch.nn as nn
from src.ncrna.models import register_model
from omegaconf import OmegaConf
from src.ncrna.models.lm.model_utils import LoRAConfig, NetConfig, get_net
from src.ncrna.sampling.sampling import ancestral_sample, optimize_sample
from pprint import pprint


@dataclass
class EvoFlowConfig:
    num_diffusion_timesteps: int = field(
        default=500
    )
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig())
    net: NetConfig = field(default_factory=lambda: NetConfig())
    gradient_ckpt: bool = field(
        default=False
    )
    rdm_couple: bool = field(
        default=False
    )

@register_model('rnadiff')
class EvoFlow(nn.Module):
    _default_cfg = EvoFlowConfig()

    def __init__(self, cfg, net=None):
        super().__init__()
        self._update_cfg(cfg)
        
        self.net = get_net(cfg) if net is None else net
        self.alphabet = self.net.alphabet
        self.mask_id = self.alphabet.mask_idx
        self.pad_id = self.alphabet.pad_idx
        self.bos_id = self.alphabet.cls_idx
        self.eos_id = self.alphabet.eos_idx
        self.x_id = self.alphabet.unk_idx
    
    def _update_cfg(self, cfg):
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)


    def forward(self, input_ids, return_last_hidden_state=False, **kwargs):
        outputs = self.net( # Modified ESM2 to output a probability distribution (embedding dim = Token space)
            input_ids=input_ids,
        )
        logits = outputs['logits']
        if return_last_hidden_state:
            last_hidden_state = outputs['last_hidden_state']
            return logits, last_hidden_state
        else:
            return logits

    def decode(self, xt, tokenizer):
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
        
        for sequence in xt:
            decoded_sequence = ''.join([tokenizer.get_tkn(idx) for idx in sequence[1:-1]])
            decoded_strings.append(decoded_sequence)
        
        return decoded_strings
    
    def sample(self, xt, steps, tau):
        # Perform sampling
        sampled_xt = ancestral_sample(
            xt=xt,
            model=self,
            tokenizer=self.alphabet,
            num_steps=steps,
            tau=tau,
            kappa_fn=lambda t: t
        )
        decoded_seqs = self.alphabet.decode(sampled_xt)
        pprint(decoded_seqs)
    
    def sample_optimized(self, xt, steps, tau):
        # Perform sampling
        sampled_xt = optimize_sample(
            xt=xt,
            model=self,
            tokenizer=self.alphabet,
            num_steps=steps,
            tau=tau,
            kappa_fn=lambda t: t
        )
        decoded_seqs = self.alphabet.decode(sampled_xt)
        pprint(decoded_seqs)

    @classmethod
    def load_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from typing import OrderedDict
        ckpt = torch.load(pretrained_model_name_or_path)
        sd = OrderedDict()
        state_dict = ckpt['state_dict']
        for k, _ in state_dict.items():
            if k[:6] == 'model.':
                sd[k[6:]] = state_dict[k]
        conf2 = OmegaConf.load('configs/lm/evoflow.yaml')
        conf2['model'].pop('_target_')
        model = cls(conf2['model'])
        model.load_state_dict(sd)
        return model

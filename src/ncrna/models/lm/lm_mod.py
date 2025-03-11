
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright (C) 2024 Atom Bioworks

import torch
import torch.nn as nn
from src.rinalmo.pretrained import get_pretrained_model
from src.rinalmo.config import model_config
from src.rinalmo.model.model import RiNALMo
from src.rinalmo.data.alphabet import Alphabet


class LM_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        # self.rm, self.alphabet = get_pretrained_model(model_name='micro-v1', lm_config='micro')#fm.pretrained.rna_fm_t12()
        config = model_config('micro')
        self.rm = RiNALMo(config)
        self.alphabet = Alphabet(**config['alphabet'])
        self.pad_id = self.alphabet.pad_idx
        self.bos_id = self.alphabet.cls_idx
        self.eos_id = self.alphabet.eos_idx

    def forward(self,
                input_ids,
                attention_mask=None,
                inputs_embeds=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
            ):
        output = self.rm(
            input_ids,
        )

        result = {
            "logits": output['logits'],
            "last_hidden_state": output['representation'],
        }
        return result
    
    def forward_encoder(self, batch, **kwargs):
        return {}
    
    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) &
            output_tokens.ne(self.bos_id) &
            output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask
    
    def initialize_output_tokens(self, batch, encoder_out, partial_masks=None, **kwargs):
        tokens = batch['input_ids']
        if tokens is None:
            raise NotImplementedError
        else:
            output_mask = self.get_non_special_sym_mask(tokens, partial_masks=partial_masks)

            output_tokens = tokens.masked_fill(output_mask, self.mask_id)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores

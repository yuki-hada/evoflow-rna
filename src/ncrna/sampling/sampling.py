import torch
from tqdm import tqdm


def stochastic_sample_from_categorical(logits, temperature=1.0, noise_scale=1.0):
    logits = logits.double()
    if temperature != 0:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        logits = logits / temperature + noise_scale * gumbel_noise
    scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores

def parse_prompt(
    prompt, 
    B, 
    tokenizer, 
    device, 
    seq_len, 
    add_special_tokens=True
):
    if prompt:
        prompt = list(prompt)
        new_prompt = ['<mask>' if x == '.' else x for x in prompt]
        seqs = [[tokenizer.cls_idx] + [tokenizer.get_idx(i) for i in new_prompt] + [tokenizer.eos_idx]]*B
    else:
        seqs = [[tokenizer.cls_idx] + [tokenizer.mask_idx]*seq_len + [tokenizer.eos_idx]]*B # full masking
    return torch.tensor(seqs, dtype=torch.int64).to(device)


@torch.inference_mode()
@torch.cuda.amp.autocast()
def ancestral_sample(xt, model, tokenizer, num_steps, tau=0.5, kappa_fn=lambda t: t):
    """
    The main ancestral sampling routine.

    Args:
        xt (Tensor): Initial tokens.
        model (callable): A model function that maps tokens -> logits.
        tokenizer (Tokenizer): A tokenizer containing at least a mask_idx attribute.
        num_steps (int): Number of iterative refinement steps.
        tau (float): Temperature used in gumbel_softmax.
        kappa_fn (callable): A function of t in [0,1] controlling the unmasking schedule.
    
    Returns:
        xt (Tensor): The final set of tokens after sampling.
    """
    dt = 1 / num_steps
    # Positions that are originally fixed
    fix_mask = xt != tokenizer.mask_idx

    # A baseline for how many we might unmask each iteration
    k = ((~fix_mask).sum(-1).float().mean() / num_steps).ceil().int().item()

    for i in range(1, num_steps + 1):
        # Identify current mask positions
        mask_t = (xt == tokenizer.mask_idx)
        # We do not change positions that are fixed or currently unmasked
        fix_mask_t = fix_mask | (~mask_t)

        # Get model logits and compute probabilities
        logits = model(xt)
        x0, score = stochastic_sample_from_categorical(logits, temperature=tau)

        x0[fix_mask_t] = xt[fix_mask_t]

        masked_score = score.masked_fill(fix_mask_t, float('-inf'))
        unfinished = (mask_t.sum(1, keepdim=True) != 0)
        if unfinished.sum() == 0:
            break
        topk_scores, topk_indices = masked_score.topk(k, dim=-1)
        unmask = torch.zeros_like(masked_score, dtype=torch.bool)
        unmask = unmask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_indices, dtype=torch.bool))
        unmask = unmask & unfinished
        xt[unmask] = x0[unmask]

        print(xt[0])

    remaining_mask = (xt == tokenizer.mask_idx)
    print(f'remaining mask: {remaining_mask.sum()}')
    xt[remaining_mask] = x0[remaining_mask]

    return xt

from torch.cuda.amp import autocast  # ← これが必須

@torch.inference_mode()
@torch.cuda.amp.autocast() 
def optimize_sample(xt, model, tokenizer, num_steps, tau=0.5, kappa_fn=lambda t: t):
    '''Ancestral sampling allowing full sequence optimization under fp16.''' 
    # ensure model and relevant layers run in half precision
    #model = model.half()
    dt = 1 / num_steps
    # Fixed tokens: keep CLS and EOS unchanged
    fix_mask = (xt == tokenizer.cls_idx) | (xt == tokenizer.eos_idx)

    # Baseline for how many tokens to update per iteration
    k = ((~fix_mask).sum(-1).float().mean() / num_steps).ceil().int().item()
    k = 8

    for i in range(1, num_steps + 1):
        # Identify current mask positions
        mask_t = (xt == tokenizer.mask_idx)
        # Do not update fixed or already unmasked positions
        fix_mask_t = fix_mask | (~mask_t)

        # Model prediction under autocast for fp16 support
        with autocast():
            logits = model(xt)
        x0, score = stochastic_sample_from_categorical(logits, temperature=tau)
        # Preserve fixed tokens
        x0[fix_mask_t] = xt[fix_mask_t]

        # Rank by confidence for masking
        masked_score = score.masked_fill(fix_mask_t, float('-inf'))
        print(masked_score)
        unfinished = (mask_t.sum(1, keepdim=True) != 0)
        #if unfinished.sum() == 0:
        #    break

        # Select top-k positions to unmask
        topk_scores, topk_indices = masked_score.topk(k, dim=-1)
        unmask = torch.zeros_like(masked_score, dtype=torch.bool)
        unmask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_indices, dtype=torch.bool))
        unmask &= unfinished

        # Update tokens
        xt[unmask] = x0[unmask]
        #print(xt[0])

    # Final fill for any remaining mask positions
    remaining_mask = (xt == tokenizer.mask_idx)
    print(f'remaining mask: {remaining_mask.sum()}')
    xt[remaining_mask] = x0[remaining_mask]

    return xt

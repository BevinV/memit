from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .memit_hparams import MEMITHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    hidden_size = getattr(model.config, "n_embd", None) or getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError("Model config has neither 'n_embd' nor 'hidden_size'")
    delta = torch.zeros((hidden_size,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # cur_out[0] has shape [batch_size, seq_len, hidden]
            seq_len = cur_out[0].shape[1]

            # Store initial value of the vector of interest
            if target_init is None:
                # clamp first index too
                safe_idx0 = min(lookup_idxs[0], seq_len - 1)
                print("Recording initial value of v*")
                target_init = cur_out[0][0, safe_idx0].detach().clone()

            # Add intervened delta, clamping each idx
            for i, raw_idx in enumerate(lookup_idxs):
                safe_idx = min(raw_idx, seq_len - 1)
                cur_out[0][i, safe_idx, :] += delta

        return cur_out
    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            : len(rewriting_prompts)
        ]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module, given fully‐formatted contexts.
    """

    # Build the layer name
    layer_name = module_template.format(layer)

    # Tokenize all prompts at once (no special tokens)
    tok_out = tok(context_templates, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = tok_out["input_ids"].to(model.device)
    attention_mask = tok_out["attention_mask"].to(model.device)

    # Find subject lookup indices for each prompt
    lookup_idxs = [
        find_fact_lookup_idx(prompt, subject, tok, fact_token_strategy, verbose=(i == 0))
        for i, (prompt, subject) in enumerate(zip(context_templates, words))
    ]

    # Run a single forward to grab input/output at that layer
    with nethook.TraceDict(
        module=model,
        layers=[layer_name],
        retain_input=True,
        retain_output=True,
    ) as tr:
        _ = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Extract input and output representations
    reps = tr[layer_name]
    # reps.input and reps.output have shape [batch, seq_len, hidden]
    batch_size, seq_len, hidden = reps.input.shape

    # Clamp indices to seq_len-1
    safe_idxs = [min(idx, seq_len - 1) for idx in lookup_idxs]

    # Gather the vectors at each index
    # Shape will be [batch, hidden]
    l_input = reps.input[range(batch_size), safe_idxs, :].detach()
    l_output = reps.output[range(batch_size), safe_idxs, :].detach()

    return l_input, l_output



def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a fully-formatted sentence
    and the subject string, by finding the subject's token span.
    Falls back to last token if not found.
    """

    # Tokenize the full prompt (no special tokens)
    input_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0].tolist()
    # Tokenize the subject (no special tokens)
    subj_ids = tok(subject, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0].tolist()

    # Try to locate subject subtokens
    for i in range(len(input_ids) - len(subj_ids) + 1):
        if input_ids[i : i + len(subj_ids)] == subj_ids:
            idx = i + len(subj_ids) - 1
            if verbose:
                tok_str = tok.decode([input_ids[idx]])
                print(f"Lookup index found: {idx} | Prompt: {prompt!r} | Token: {tok_str!r}")
            return idx

    # --- Fallback: use last token ---
    idx = len(input_ids) - 1
    if verbose:
        print(
            f"WARNING: subject {subject!r} not found in prompt tokens, "
            f"defaulting lookup index to last token ({idx}).\n"
            f"Full prompt: {prompt!r}"
        )
    return idx


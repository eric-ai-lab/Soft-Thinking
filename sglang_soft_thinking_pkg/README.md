# Soft Thinking: Modifications Overview

This document summarizes the differences between this fork and the original sglang (v0.4.6.post1). For a detailed diff, please refer to `changes_0.4.6.post1.diff`.

All added or modified code is clearly marked with:
```python
# ==========
# begin of soft thinking
# ==========
```
and
```python
# ==========
# end of soft thinking
# ==========
```

---

# Main Strategies in Soft Thinking

The core strategies of Soft Thinking are implemented in:
- `sglang_soft_thinking_pkg/python/sglang/srt/layers/sampler.py`
- `sglang_soft_thinking_pkg/python/sglang/srt/layers/vocab_parallel_embedding.py`
- `sglang_soft_thinking_pkg/python/sglang/srt/models/`

## 1. Sampler Logic

We first apply top-k, top-p, and min-p filtering, then sample for indices that have completed "thinking". The relevant code snippet is as follows:

```python
                    # calculate the entropy
                    entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)
                    soft_mask = sampling_info.soft_thinking_modes # Shape (B,)
                    top_ps = torch.where(soft_mask, sampling_info.top_ps, sampling_info.after_thinking_top_ps)
                    top_ks = torch.where(soft_mask, sampling_info.top_ks, sampling_info.after_thinking_top_ks)
                    min_ps = torch.where(soft_mask, sampling_info.min_ps, sampling_info.after_thinking_min_ps)
                    dirichlet_alphas = sampling_info.dirichlet_alphas

                    # top k top p renorm
                    probs = top_k_renorm_prob(probs, top_ks)
                    probs = top_p_renorm_prob(probs, top_ps)

                    # minp renorm
                    if sampling_info.need_min_p_sampling or sampling_info.need_after_thinking_min_p_sampling: # slow
                        max_prob = probs.max(dim=-1, keepdim=True).values
                        min_p_thresholds = max_prob * min_ps.view(-1, 1)
                        min_p_mask = probs < min_p_thresholds
                        probs.masked_fill_(min_p_mask, 0.0)
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                    # dirichlet noise (not used in paper)
                    if not sampling_info.is_all_no_noise: # slow
                        conc = probs[soft_mask] * dirichlet_alphas[soft_mask].view(-1, 1)
                        gamma_dist = torch.distributions.Gamma(conc, torch.ones_like(conc))
                        gamma_samples = gamma_dist.sample()
                        probs_new = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
                        probs[soft_mask] = probs_new

                    # max top k
                    topk_probs, topk_indices = torch.topk(probs, k=sampling_info.max_topk, dim=-1) # slow
                    topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True))

                    # after thinking sampling
                    non_soft_mask = ~soft_mask
                    if any(non_soft_mask):
                        sampled_token_ids = torch.multinomial(probs, num_samples=1)

                        # For rows where soft_thinking_modes is False
                        topk_probs[non_soft_mask] = 0.0
                        topk_indices[non_soft_mask] = 0

                        # Assign the first element of each row to sampled_token_ids and set it to 1.0 in topk_probs
                        topk_probs[non_soft_mask, 0] = 1.0
                        topk_indices[non_soft_mask, 0] = sampled_token_ids[non_soft_mask].view(-1)

                    logits_output.topk_probs = topk_probs
                    logits_output.topk_indices = topk_indices
                    logits_output.entropy = entropy
                    batch_next_token_ids = topk_indices[:, 0].to(torch.int32)
```

## 2. New Embedding Calculation

For Soft Thinking, the new embedding is computed as a weighted sum of the top-k token embeddings:

$$
\text{new\_embedding} = \sum_{i=1}^{k} \text{topk\_probs}_i \times \text{Embedding}[\text{topk\_{indices}}_i]
$$

```python
    # topk_probs is not None and topk_indices
    def weighted_forward(self, topk_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """Single-GPU weighted embedding forward.

        Args:
            topk_probs: [B, K] tensor of probabilities for top-K tokens.
            topk_indices: [B, K] tensor of token indices for top-K tokens.

        Returns:
            hidden_states: [B, D] weighted embedding.
        """

        # Validate inputs
        assert topk_probs.shape == topk_indices.shape, "topk_probs and topk_indices must have same shape."

        # Use quant_method.embedding for consistency
        topk_embeddings = self.quant_method.embedding(self, topk_indices.long())  # [B, K, D]
        # Normalize probs to sum to 1.0 along last dim.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True) # do norm here
        hidden_states = torch.sum(topk_probs.unsqueeze(-1) * topk_embeddings, dim=1, dtype=topk_embeddings.dtype)  # [B, D]
        return hidden_states

    def weighted_forward_tp(self, topk_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """Tensor Parallel weighted embedding forward.

        Args:
            topk_probs: [B, K] tensor of probabilities for top-K tokens.
            topk_indices: [B, K] tensor of token indices for top-K tokens.

        Returns:
            hidden_states: [B, D] weighted embedding after TP all-reduce.
        """
        # Validate inputs
        assert topk_probs.shape == topk_indices.shape, "topk_probs and topk_indices must have same shape."
        # Normalize probs to sum to 1.0 along last dim.
        # topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        masked_indices, input_mask = self.get_masked_indices_and_mask(
            topk_indices,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
        )
        topk_embeddings: torch.Tensor = self.quant_method.embedding(self, masked_indices.long())  # [B, K, D]
        input_mask = input_mask.unsqueeze(-1)  # [B, K, 1]
        topk_embeddings.masked_fill_(input_mask, 0)  # Zero out invalid indices
        hidden_states_parallel = torch.sum(
            topk_probs.unsqueeze(-1) * topk_embeddings, dim=1, dtype=topk_embeddings.dtype
        )  # [B, D]
        hidden_states = tensor_model_parallel_all_reduce(hidden_states_parallel)
        return hidden_states
    def get_masked_indices_and_mask(
        self,
        indices: torch.Tensor,
        org_vocab_start_index: int,
        org_vocab_end_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map indices to current GPU's vocab shard and generate mask.

        Args:
            indices: [B, K] tensor of token indices.
            org_vocab_start_index: Start index of current GPU's vocab shard.
            org_vocab_end_index: End index of current GPU's vocab shard.

        Returns:
            masked_indices: [B, K] mapped indices for current shard.
            vocab_mask: [B, K] boolean mask (True for invalid indices).
        """
        vocab_mask = (indices >= org_vocab_start_index) & (indices < org_vocab_end_index)
        valid_offset = org_vocab_start_index * vocab_mask
        masked_indices = vocab_mask * (indices - valid_offset)
        return masked_indices, ~vocab_mask
```

## 3. Model Forward Pass

The new forward logic for Soft Thinking is implemented in `qwen2.py` and `llama.py`. To support additional models, copy the following code into the model's `forward` method and adapt as needed:

```python
        # ==========
        # begin of soft thinking
        # ==========
        if forward_batch.topk_probs is not None and forward_batch.topk_indices is not None:
            if self.tp_size > 1:
                hidden_states = self.embed_tokens.weighted_forward_tp(
                    forward_batch.topk_probs, forward_batch.topk_indices
                )
            else:
                hidden_states = self.embed_tokens.weighted_forward(
                    forward_batch.topk_probs, forward_batch.topk_indices
                )  
        elif input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        # ==========
        # end of soft thinking
        # ==========
```

---

# Unchanged Aspects

We have implemented CUDA graph support, but have not implemented overlap.

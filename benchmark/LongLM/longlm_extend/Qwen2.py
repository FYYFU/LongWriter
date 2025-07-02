# Copied from the implementation for Mistral.
# transfromers version 4.38.2
# No support of sliding window. Check our paper for more reason about why we don't use it.
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from transformers.cache_utils import Cache
import numpy as np
# from flash_attn import flash_attn_func, flash_attn_varlen_func
# from .selfextend_flash_attn import self_extend_flash_forward


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_longlm_rotary_pos_emb(q, k, cos, sin, position_ids, group_size=1, window_size=4096):

    import ipdb
    cos2d = cos.squeeze(1).squeeze(0)
    sin2d = sin.squeeze(1).squeeze(0)

    pos_q = position_ids # [batch, q_len]
    pos_k = torch.arange(cos2d.shape[0], device=cos2d.device, dtype=position_ids.dtype)
    pos_k = pos_k.view(1, -1) # [1, kv_len]

    window_size = 0 if position_ids.max() < window_size else window_size
    pos_q_g = pos_q // group_size + window_size - window_size // group_size
    pos_k_g = pos_k // group_size

    def idx(t2d, pos):
        return t2d[pos].unsqueeze(1)

    # neighbor (window)
    cos_n_q = idx(cos2d, pos_q)
    sin_n_q = idx(sin2d, pos_q)
    cos_n_k = idx(cos2d, pos_k)
    sin_n_k = idx(sin2d, pos_k)

    # group
    cos_g_q = idx(cos2d, pos_q_g)
    sin_g_q = idx(sin2d, pos_q_g)
    cos_g_k = idx(cos2d, pos_k_g)
    sin_g_k = idx(sin2d, pos_k_g)

    nq = (q * cos_n_q) + (rotate_half(q) * sin_n_q)
    nk = (k * cos_n_k) + (rotate_half(k) * sin_n_k)
    gq = (q * cos_g_q) + (rotate_half(q) * sin_g_q)
    gk = (k * cos_g_k) + (rotate_half(k) * sin_g_k)

    return nq, nk, gq, gk


def longlm_eager_attention_forward(
    module: nn.Module,
    neighbor_query: torch.Tensor,
    neighbor_key: torch.Tensor,
    group_query: torch.Tensor,
    group_key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    window_size: int = 2048,
    group_size: int = 2,
    **kwargs,
):
    neighbor_key_states = repeat_kv(neighbor_key, module.num_key_value_groups)
    group_key_states = repeat_kv(group_key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    
    neighbor_attn_weights = torch.matmul(neighbor_query, neighbor_key_states.transpose(2, 3)) * scaling
    group_attn_weights = torch.matmul(group_query, group_key_states.transpose(2, 3)) * scaling

    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : group_key_states.shape[-2]]
        neighbor_attn_weights = neighbor_attn_weights + causal_mask
        group_attn_weights = group_attn_weights + causal_mask

    q_len = group_query.shape[-2]
    kv_seq_len = group_key.shape[-2]

    if group_query.shape[-2] == 1:
        neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask[:, -window_size:] = 1
    elif q_len == kv_seq_len:
        neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask = torch.tril(neighbor_attention_mask)
        if q_len - window_size > 0:
            group_attention_mask =  torch.tril(torch.ones((q_len - window_size, kv_seq_len - window_size), device=group_attn_weights.device))
            neighbor_attention_mask[window_size:, :-window_size] -= group_attention_mask
    else:
        raise ValueError("q_len should be 1 or seq_len.")


    neighbor_attention_mask = neighbor_attention_mask.bool()
    attn_weights = torch.where(neighbor_attention_mask, neighbor_attn_weights, group_attn_weights)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(group_query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from typing import Callable, Optional, Union


def longlm_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    group_size: Optional[int] = 8,
    window_size: Optional[int] = 2048,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # cos, sin = position_embeddings
    # 现在的主要问题是 现在的cos似乎只计算一个位置
    # 但是在之前的版本里面，cos并不是计算某一个位置的，而是所有的位置都计算了
    ####!
    position_ids = kwargs['position_ids']
    kv_seq_len = key_states.shape[-2]
    q_len = query_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    if q_len  == 1:
        position = torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_states.device).view(1, kv_seq_len) # only support batch=1 for now.
        cos, sin = self.rotary_emb(value_states, position)
    else:
        cos, sin = self.rotary_emb(value_states, position_ids)
    ####!

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    #! 这里也进行了修改，保存的key和value放在了旋转之前
    neighbor_query_states, neighbor_key_states, \
    group_query_states, group_key_states = apply_longlm_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids=position_ids,
        group_size=group_size, window_size=window_size
    )
    attention_interface: Callable = longlm_eager_attention_forward
    if self.config._attn_implementation != "eager":
        raise ValueError('Current only support eager')
        # attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        neighbor_query_states,
        neighbor_key_states,
        group_query_states,
        group_key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # main diff with Llama
        window_size=window_size,
        group_size=group_size,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
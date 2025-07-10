# Copied from the implementation for Mistral.
# transfromers version 4.38.2
# No support of sliding window. Check our paper for more reason about why we don't use it.
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from transformers.cache_utils import Cache
import numpy as np
from flash_attn import flash_attn_func, flash_attn_varlen_func
# from .selfextend_flash_attn import self_extend_flash_forward

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

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

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos[:,:, -q.shape[2]:]) + (rotate_half(q) * sin[:,:, -q.shape[2]:]) if q is not None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None
    return q_embed, k_embed

def apply_longlm_rotary_pos_emb(q, k, cos, sin, position_ids, key_position_ids, group_size=1, window_size=4096):

    cos2d = cos.squeeze(1).squeeze(0)
    sin2d = sin.squeeze(1).squeeze(0)

    pos_q = position_ids # [batch, q_len]
    # pos_k = torch.arange(cos2d.shape[0], device=cos2d.device, dtype=position_ids.dtype)
    # pos_k = pos_k.view(1, -1) # [1, kv_len]
    pos_k = key_position_ids

    re_window_size = 0 if position_ids.max() < window_size else window_size
    pos_q_g = pos_q // group_size + re_window_size - re_window_size // group_size
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
from flash_attn.bert_padding import unpad_input as unpad_input, pad_input
from transformers.modeling_flash_attention_utils import _upad_input

def flash_attention2_forward_with_window_size(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    window_size=[-1, -1],
    return_attn_probs=False,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        window_size ([Int, Int])
            The left & right window size for Flash Attention. Default to [-1, -1] which means no window size is used.
        return_attn_probs (`bool`, *optional*):
            Whether to return the attention softmax logssumexp and probabilities. Default to False.
    """
    causal = True

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, unpad_input
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        attn_output_unpad, softmax_lse, S_dmask = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_attn_probs=True,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output, softmax_lse, S_dmask = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_attn_probs=True,
        )

    if return_attn_probs:
        return attn_output, softmax_lse, S_dmask
    else:
        return attn_output

def longlm_flash_forward(
        model_self,
        query_position,
        group_size_2,
        neighbor_query_states,
        neighbor_key_states,
        group_query_states,
        group_key_states,
        value_states,
        attention_mask,
        bsz,
        q_len,
        kv_seq_len,
        attn_dropout,
    ):
    
    if query_position.max() >= group_size_2:
        neighbor_attn_output, neighbor_softmax_lse_right_padded, neighbor_prob = flash_attention2_forward_with_window_size(
            neighbor_query_states,
            neighbor_key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            window_size=[group_size_2 - 1, 0],
            # right dim here does not matter and can be -1, or > 0 due to causal mask
            return_attn_probs=True,
        )

        group_attention_len = (
            kv_seq_len - group_size_2
        )  # here we should use kv_seq_len rather than max_kv_len since we have paddings in qkv and attention_mask

        group_attention_mask = attention_mask[:, :group_attention_len] if not attention_mask is None else None
        group_attn_output, group_softmax_lse_right_padded, group_prob = flash_attention2_forward_with_window_size(
            group_query_states[:, -group_attention_len:, :, :],
            group_key_states[:, :group_attention_len, :, :],
            value_states[:, :group_attention_len, :, :],
            group_attention_mask,
            group_query_states[:, -group_attention_len:, :, :].shape[1],
            dropout=attn_dropout,
            window_size=[-1, -1],
            return_attn_probs=True,
        )  # note that kv and q's indexing are different! also query size could be different from kv length and very small during generation compared to prefilling


        # normalize lse first
        neighbor_seq_length = torch.Tensor([kv_seq_len,]).long().expand(bsz, 1) if attention_mask is None else torch.sum(attention_mask, axis=1, keepdim=True)  # [batch_size, 1]
        group_seq_length = torch.Tensor([group_attention_len,]).long().expand(bsz, 1) if attention_mask is None else torch.sum(attention_mask[:, :group_attention_len], axis=1, keepdim=True)  # [batch_size, 1]

        # convert align left to align right and convert exp(0) to 0
        neighbor_softmax_lse = torch.zeros_like(neighbor_softmax_lse_right_padded)
        group_softmax_lse = torch.zeros_like(group_softmax_lse_right_padded)
        for idx in range(bsz):
            if neighbor_seq_length[idx] > 0:
                neighbor_softmax_lse[idx, :, -neighbor_seq_length[idx] :] = neighbor_softmax_lse_right_padded[
                    idx, :, : neighbor_seq_length[idx]
                ]
            if group_seq_length[idx] > 0:
                group_softmax_lse[idx, :, -group_seq_length[idx] :] = group_softmax_lse_right_padded[
                    idx, :, : group_seq_length[idx]
                ]

        # attn_output size is [batch_size, max_seq_len (not the true one), query_length, dim]
        true_neighbor_seq_max_length = neighbor_softmax_lse.shape[
            -1
        ]  # it could be smaller than query_length due to the attention_mask
        true_group_seq_max_length = group_softmax_lse.shape[
            -1
        ]  # it could be smaller than group_query_layer[:, -group_attention_len:, :, :].shape[1] due to the attention_mask[:, :group_attention_len]

        neighbor_softmax_lse = neighbor_softmax_lse.transpose(1, 2).unsqueeze(
            -1
        )  # [batch_size, true_neighbor_seq_max_length, self.num_heads, 1]
        group_softmax_lse = group_softmax_lse.transpose(1, 2).unsqueeze(
            -1
        )  # [batch_size, true_group_seq_max_length, self.num_heads, 1]

        lse_gap = group_softmax_lse - neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :]
        #if  torch.isinf(neighbor_softmax_lse).any() or torch.isnan(neighbor_softmax_lse).any():
        #    import pdb; pdb.set_trace()
        
        neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :] = 1 / (1 + torch.exp(lse_gap))
        neighbor_softmax_lse[:, :-true_group_seq_max_length, :, :] = 1.
        group_softmax_lse = 1 / (1 + torch.exp(-lse_gap))



        neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] = (
            neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] * neighbor_softmax_lse
        )
        group_attn_output[:, -true_group_seq_max_length:, ...] = (
            group_attn_output[:, -true_group_seq_max_length:, ...] * group_softmax_lse
        )
        attn_output = torch.empty_like(neighbor_attn_output).copy_(
            neighbor_attn_output
        )  # might be slightly faster than clone
        #attn_output[:, group_size_2:, ...] += group_attn_output
        attn_output[:, group_size_2-kv_seq_len:, ...] += group_attn_output
        attn_output = torch.nan_to_num(attn_output, nan=0)  
    
    else:
        attn_output = flash_attention2_forward_with_window_size(
            neighbor_query_states,
            neighbor_key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            window_size=[-1, -1],
        )

    return attn_output


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

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # cos, sin = position_embeddings
    # 现在的主要问题是 现在的cos似乎只计算一个位置. 但是在之前的版本里面，cos并不是计算某一个位置的，而是所有的位置都计算了
    ####!
    position_ids = kwargs['position_ids']
    kv_seq_len = key_states.shape[-2]
    bsz, q_len = query_states.shape[0], query_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    if q_len  == 1:
        # key_position = torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_states.device).view(1, kv_seq_len) # only support batch=1 for now.
        key_position = self.max_position_ids[:, :kv_seq_len]
        cos, sin = self.rotary_emb(value_states, key_position)
    else:
        key_position = position_ids
        cos, sin = self.rotary_emb(value_states, position_ids)
    cos = cos.to(query_states.device)
    sin = sin.to(query_states.device)
    ####!


    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = longlm_eager_attention_forward
    if self.config._attn_implementation == "eager":
        neighbor_query_states, neighbor_key_states, \
        group_query_states, group_key_states = apply_longlm_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids=position_ids, key_position_ids=key_position,
            group_size=group_size, window_size=window_size
        )
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
    elif self.config._attn_implementation == "flash_attention_2":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_dropout = self.config.attention_dropout if self.training else 0.0
        attn_weights = None

        if q_len == 1:
            re_window_size = 0 if position_ids.max() < window_size else window_size
            neighbor_key_position = position_ids[:, -1] - key_position
            group_key_position = position_ids[:, -1]//group_size - key_position//group_size + (re_window_size - re_window_size//group_size)
            decode_key_position = torch.cat([group_key_position[:, :-window_size], neighbor_key_position[:,-window_size:]], dim=1)
                
            decode_query_states = query_states.transpose(1,2).contiguous() # position 0: cos 0 = 1, sin 0 = 0
            _, decode_key_states = apply_rotary_pos_emb(None, key_states, cos, -sin, decode_key_position) 

            decode_key_states = repeat_kv(decode_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
            decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
            
            attn_output = flash_attn_func(decode_query_states,
                                        decode_key_states,
                                        decode_value_states,
                                        attn_dropout, 
                                        softmax_scale=None, 
                                        causal=True)
    
        elif q_len == kv_seq_len:
            # set correct position_ids & apply RoPE.
            # window_size = 0 if position_ids.max() < window_size else window_size # in case that, the smallest q position, g2-g2//g1 exceed the max position
            
            neighbor_query_states, neighbor_key_states, \
            group_query_states, group_key_states = apply_longlm_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids=position_ids, key_position_ids=key_position,
                group_size=group_size, window_size=window_size
            )
            neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
            neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
            group_query_states = group_query_states.transpose(1, 2).contiguous()
            group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
            value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()

            attn_output = longlm_flash_forward(self,
                                                position_ids,
                                                window_size,
                                                neighbor_query_states,
                                                neighbor_key_states,
                                                group_query_states,
                                                group_key_states,
                                                value_states,
                                                attention_mask,
                                                bsz,
                                                q_len,
                                                kv_seq_len,
                                                attn_dropout,
                                            )
        else:
            raise ValueError("q_len should be 1 or seq_len.")
    else:
        raise ValueError('Current only support eager & flash-attn-2')

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


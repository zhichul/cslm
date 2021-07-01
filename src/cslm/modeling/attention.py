import torch
from torch import nn as nn

from cslm.modeling.module import Module


def causal_mask(max_positions):
    return torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8))


class MultiheadAttentionBase(Module):

    def __init__(self, config):
        super().__init__()
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer("mask_out_value", torch.tensor(-1e4))

    def split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits n_embd dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_embd
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def attn_weights(self, query, key, mask=None):
        """
        Computes
        :param query: (batch, head, seq_length, query_size)
        :param key:   (batch, head, seq_length,   key_size)
        :param value: (batch, head, seq_length, value_size)
        :param mask:  (    1,    1, seq_length, seq_length) can be integer or boolean
        :return:
        """
        # compute QK^T
        attn_scores = torch.matmul(query, key.transpose(-1, -2))

        # scale by sqrt{d_k}
        attn_scores = attn_scores / (float(key.size(-1)) ** 0.5)

        # mask it by adding mask_out_value (large negative number) to all positions that should not be attended
        if mask is not None:
            attn_scores = torch.where(mask.bool(), attn_scores, self.mask_out_value.to(attn_scores.dtype))

        attn_weights = nn.Softmax(dim=-1)(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)
        return self.expose(attn_weights, "attn_weights")

    def attn_output(self, value, attn_weights):
        """

        :param value:        (batch, head, seq_length, value_size)
        :param attn_weights: (batch, head, seq_length, seq_lenth) -2 indexes attn source, -1 indexes attn target
        :return:
        """
        attn_output = torch.matmul(attn_weights, value)
        return self.expose(attn_output, "attn_output")


class MultiheadUnmixedAttention(MultiheadAttentionBase):

    def __init__(self, config):
        super().__init__(config)

        self.n_embd = config.n_embd
        self.num_heads = config.n_head
        self.n_qk = config.n_qk
        self.n_v = config.n_v

        self.q = nn.Linear(self.n_embd, self.n_qk * self.num_heads)
        self.k = nn.Linear(self.n_embd, self.n_qk * self.num_heads)
        self.v = nn.Linear(self.n_embd, self.n_v * self.num_heads)

    def forward(
            self,
            q_features=None,
            kv_features=None,
            mask=None
    ):
        """

        :param hidden_states:           (batch, dec_length, n_embd)
        :param encoder_hidden_states:   (batch, enc_length, n_embd)
        :param encoder_attention_mask:  (batch, enc_length)
        :return:
        """

        # compute q,k,v
        query = self.q(q_features)
        key = self.k(kv_features)
        value = self.v(kv_features)
        attention_mask = mask

        # split into heads for multihead attention
        query = self.split_heads(query, self.num_heads, self.n_qk)
        key = self.split_heads(key, self.num_heads, self.n_qk)
        value = self.split_heads(value, self.num_heads, self.n_v)

        # scaled dot product attention
        attn_weights = self.attn_weights(query, key, attention_mask)
        attn_output = self.attn_output(value, attn_weights)

        attn_output = self.merge_heads(attn_output, self.num_heads, self.n_v)

        # just dropout no projection to mix things
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class TransformerMultiheadAttention(MultiheadAttentionBase):

    def __init__(self, config):
        super().__init__(config)

        self.n_embd = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.n_embd // self.num_heads
        self.validate_dimensions()

        self.kv = nn.Linear(self.n_embd, 2 * self.n_embd)
        self.q = nn.Linear(self.n_embd, self.n_embd)
        self.proj = nn.Linear(self.n_embd, self.n_embd)

    def validate_dimensions(self):
        if self.head_dim * self.num_heads != self.n_embd:
            raise ValueError(
                f"`n_embd` must be divisible by `num_heads` (got `n_embd`: {self.n_embd} and `num_heads`: {self.num_heads})."
            )

    def forward(
        self,
        q_features=None,
        kv_features=None,
        mask=None
    ):
        """

        :param hidden_states:           (batch, dec_length, n_embd)
        :param encoder_hidden_states:   (batch, enc_length, n_embd)
        :param encoder_attention_mask:  (batch, enc_length)
        :return:
        """

        # compute q,k,v
        query = self.q(q_features)
        key, value = self.kv(kv_features).split(self.n_embd, dim=-1)
        attention_mask = mask

        # split into heads for multihead attention
        query = self.split_heads(query, self.num_heads, self.head_dim)
        key = self.split_heads(key, self.num_heads, self.head_dim)
        value = self.split_heads(value, self.num_heads, self.head_dim)

        # scaled dot product attention
        attn_weights = self.attn_weights(query, key, attention_mask)
        attn_output = self.attn_output(value, attn_weights)

        attn_output = self.merge_heads(attn_output, self.num_heads, self.head_dim)

        # projection
        attn_output = self.proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class CrossAttention(Module):

    def __init__(self, config):
        super().__init__()
        if config.attn_type == "transformer_multihead":
            self.attention = TransformerMultiheadAttention(config)
        elif config.attn_type == "softmix_multihead":
            self.attention = MultiheadUnmixedAttention(config)
        else:
            raise ValueError(
                f"`config.attn_type` must be one of {{'transformer_multihead', 'softmix_multihead'}} (got `config.attn_type`: {config.attn_type})."
            )

    def forward(self,
                hidden_states=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        return self.attention(q_features=hidden_states,
                              kv_features=encoder_hidden_states,
                              mask=encoder_attention_mask)


class CausalSelfAttention(Module):

    def __init__(self, config):
        super().__init__()
        max_positions = config.n_positions
        self.register_buffer("causal_mask", causal_mask(max_positions).view(1, 1, max_positions, max_positions))
        if config.attn_type == "transformer_multihead":
            self.attention = TransformerMultiheadAttention(config)
        elif config.attn_type == "softmix_multihead":
            self.attention = MultiheadUnmixedAttention(config)
        else:
            raise ValueError(
                f"`config.attn_type` must be one of {{'transformer_multihead', 'softmix_multihead'}} (got `config.attn_type`: {config.attn_type})."
            )

    def forward(self,
                hidden_states=None,
                attention_mask=None):
        seq_length = attention_mask.size(-1)
        return self.attention(q_features=hidden_states,
                              kv_features=hidden_states,
                              mask=attention_mask * self.causal_mask[:,:,:seq_length, :seq_length])


class SelfAttention(Module):

    def __init__(self, config):
        super().__init__()
        max_positions = config.n_positions
        if config.attn_type == "transformer_multihead":
            self.attention = TransformerMultiheadAttention(config)
        elif config.attn_type == "softmix_multihead":
            self.attention = MultiheadUnmixedAttention(config)
        else:
            raise ValueError(
                f"`config.attn_type` must be one of {{'transformer_multihead', 'softmix_multihead'}} (got `config.attn_type`: {config.attn_type})."
            )

    def forward(self,
                hidden_states=None,
                attention_mask=None):
        return self.attention(q_features=hidden_states,
                              kv_features=hidden_states,
                              mask=attention_mask)


from cslm.modeling.activations import ACTIVATIONS
from cslm.modeling.module import Module

import torch
import torch.nn as nn



def causal_mask(max_positions):
    return torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8))


class MultiheadAttention(Module):

    def __init__(self, config):
        super().__init__()

        self.n_embd = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.n_embd // self.num_heads
        self.validate_dimensions()

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.kv = nn.Linear(self.n_embd, 2 * self.n_embd)
        self.q = nn.Linear(self.n_embd, self.n_embd)
        self.proj = nn.Linear(self.n_embd, self.n_embd)

        self.register_buffer("mask_out_value", torch.tensor(-1e4))

    def validate_dimensions(self):
        if self.head_dim * self.num_heads != self.n_embd:
            raise ValueError(
                f"`n_embd` must be divisible by `num_heads` (got `n_embd`: {self.n_embd} and `num_heads`: {self.num_heads})."
            )

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
        if config.attn_type == "multihead":
            self.attention = MultiheadAttention(config)
        else:
            raise ValueError(
                f"`config.attn_type` must be one of {{'multihead'}} (got `config.attn_type`: {config.attn_type})."
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
        if config.attn_type == "multihead":
            self.attention = MultiheadAttention(config)
        else:
            raise ValueError(
                f"`config.attn_type` must be one of {{'multihead'}} (got `config.attn_type`: {config.attn_type})."
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
        if config.attn_type == "multihead":
            self.attention = MultiheadAttention(config)
        else:
            raise ValueError(
                f"`config.attn_type` must be one of {{'multihead'}} (got `config.attn_type`: {config.attn_type})."
            )

    def forward(self,
                hidden_states=None,
                attention_mask=None):
        return self.attention(q_features=hidden_states,
                              kv_features=hidden_states,
                              mask=attention_mask)


class TransformerMLP(Module):

    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        inner_size = config.n_inner if config.n_inner is not None else 4 * n_embd
        self.fc = nn.Linear(n_embd, inner_size)
        self.proj = nn.Linear(inner_size, n_embd)
        self.act = ACTIVATIONS[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(Module):

    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd

        if config.is_encoder:
            self.ln_1 = nn.LayerNorm(n_embd, eps=config.layer_norm_epsilon)
            self.selfattention = SelfAttention(config)
        else:
            self.ln_1 = nn.LayerNorm(n_embd, eps=config.layer_norm_epsilon)
            self.selfattention = CausalSelfAttention(config)

        if config.add_cross_attention:
            self.crossattention = CrossAttention(config)
            self.ln_cross_attn = nn.LayerNorm(n_embd, eps=config.layer_norm_epsilon)

        self.ln_2 = nn.LayerNorm(n_embd, eps=config.layer_norm_epsilon)
        self.mlp = TransformerMLP(config)

    def forward(
        self,
        hidden_states=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # apply self-attention
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.selfattention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        # apply residual connection
        hidden_states = attn_output + hidden_states


        # apply cross attention
        if encoder_hidden_states is not None:
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_output = self.crossattention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )

            # apply residual connection
            hidden_states = cross_attn_output + hidden_states

        # apply mlp
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        # apply residual connection
        hidden_states = feed_forward_hidden_states + hidden_states

        return hidden_states


class Transformer(Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.n_embd = config.n_embd

        self.word_type_embed = nn.Embedding(config.vocab_size, self.n_embd)
        self.word_position_embed = nn.Embedding(config.n_positions, self.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        batch_size = input_ids.size(0)
        input_size = input_ids.size()
        device = input_ids.device


        # for tensor with arbitrary number of axis, transform to  (-1, seq_length) matrix, run the model, then
        # transform back
        input_ids = input_ids.view(-1, input_ids.size(-1))

        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.size(-1))

        attention_mask = attention_mask.view(-1,  input_ids.size(-1))[:, None, None, :] # to be broadcastable to (batch, head, seq_length, seq_length)

        # resize encoder inputs
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.view(-1, encoder_hidden_states.size(-2), encoder_hidden_states.size(-1))
            encoder_attention_mask = encoder_attention_mask.view(-1, encoder_attention_mask.size(-1))[:, None, None, :]

        # run the model
        inputs_embeds = self.word_type_embed(input_ids)
        position_embeds = self.word_position_embed(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        for i, block in enumerate(self.h):

            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        hidden_states = self.ln_f(hidden_states)

        # resize the hidden states
        output_shape = input_size + (hidden_states.size(-1),)
        hidden_states = hidden_states.view(*output_shape)

        return hidden_states

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_weights(self):
        self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.word_type_embed

from cslm.modeling.activations import ACTIVATIONS
from cslm.modeling.attention import CrossAttention, CausalSelfAttention, SelfAttention
from cslm.modeling.head import LMHead
from cslm.modeling.module import Module

import torch
import torch.nn as nn


class TransformerLMHead(LMHead):

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self,
                hidden_states=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        return self.linear(hidden_states)


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
        if self.config.pos_embd:
            self.word_position_embed = nn.Embedding(config.n_positions, self.n_embd)
        if self.config.lang_embd:
            self.language_embedding = nn.Embedding(2, self.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        language_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        batch_size = input_ids.size(0)
        input_size = input_ids.size()
        device = input_ids.device


        # for tensor with arbitrary number of axis, transform to  (-1, seq_length) matrix, run the model, then
        # transform back
        input_ids = input_ids.reshape(-1, input_ids.size(-1))

        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).reshape(-1, input_ids.size(-1))

        attention_mask = attention_mask.reshape(-1,  input_ids.size(-1))[:, None, None, :] # to be broadcastable to (batch, head, seq_length, seq_length)

        # resize encoder inputs
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.reshape(-1, encoder_hidden_states.size(-2), encoder_hidden_states.size(-1))
            encoder_attention_mask = encoder_attention_mask.reshape(-1, encoder_attention_mask.size(-1))[:, None, None, :]

        # run the model
        inputs_embeds = self.word_type_embed(input_ids)
        if self.config.pos_embd:
            position_embeds = self.word_position_embed(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            print("no pos")
            hidden_states = inputs_embeds

        # optionally add language embeddings
        if language_ids is not None and self.config.lang_embd is False:
            raise ValueError("Please make sure that lang_embd=true is specified "
                             "in transformer configuration when you'd like to use language embeddings.")
        if language_ids is not None:
            language_embeds = self.language_embedding(language_ids)
            hidden_states = hidden_states + language_embeds

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
        hidden_states = hidden_states.reshape(*output_shape)

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

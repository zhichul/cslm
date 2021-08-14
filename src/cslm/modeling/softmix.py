import torch.nn as nn
import torch
from cslm.modeling.activations import ACTIVATIONS

from cslm.modeling.attention import CrossAttention
from cslm.modeling.head import LMHead
from cslm.modeling.module import Module


class ParameterLayer(Module):

    def __init__(self, size):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(size))

    def forward(self, inputs):
        """
        if inputs has size [..., x, y, z] then outputs has
        [..., x, y] + self.value.size()

        In effect simulates applying a funciton to the last dimension of the inputs,
        but just returns a bias.
        """
        extra_dims = len(inputs.size()) - 1
        shape = extra_dims * (1,) + self.value.size()
        return torch.reshape(self.value, shape).expand(*(inputs.size()[:-1] + self.value.size()))


class BlockDiagonalProjection(Module):

    def __init__(self, n_in, n_out, n_block):
        super().__init__()
        self.l = nn.ParameterList()
        self.n_in = n_in
        self.n_block = n_block
        self.n_out = n_out
        for i in range(n_block):
            self.l.append(nn.Parameter(torch.zeros((n_out, n_in))))

    def forward(self, inputs):
        return self.forward_time_heavy(inputs)

    def forward_time_heavy(self, inputs):
        batch, seq, head, n_v = inputs.size()
        head_outputs = []
        for h in range(self.n_block):
            W = self.l[h]
            batch_combined_input = inputs[..., h, :].reshape((-1, n_v))
            flattented_output = torch.bmm(W[None, ...].expand(batch * seq, *W.size()), batch_combined_input[..., None])
            reshaped_output = flattented_output.reshape((batch, seq, 1, self.n_out)) # batch * seq, 1 * n_out  -> batch, seq, 1, n_out
            head_outputs.append(reshaped_output)
        return torch.cat(head_outputs, dim=-2)

    def forward_memory_heavy(self, inputs):
        batch, seq, head, n_v = inputs.size()
        block_diag = torch.block_diag(*self.l)
        batch_combined_and_head_combined_input = inputs.reshape((-1, head * n_v))
        flattented_output = torch.bmm(block_diag[None, ...].expand(batch * seq, *block_diag.size()), batch_combined_and_head_combined_input[..., None])
        reshaped_output = flattented_output.reshape((batch, seq, head, self.n_out)) # batch * seq, head * n_out  -> batch, seq, head, n_out
        return reshaped_output


class ProjectionTransform(Module):

    def __init__(self, n_in, n_out, n_block, shared=False, nonlinearity=None):
        super().__init__()
        self.shared = shared
        self.n_block = n_block
        self.n_in = n_in
        self.n_out = n_out
        if not self.shared:
            self.linear = nn.Linear(n_in, n_out * n_block)
        else:
            self.linear = nn.Linear(n_in, n_out)
        self.nonlinearity_type = nonlinearity
        if self.nonlinearity_type is not None:
            self.nonlinearity = ACTIVATIONS[nonlinearity]
        else:
            self.nonlinearity = lambda x: x

    def forward(self,
                hidden_states=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        if not self.shared:
            return self.nonlinearity(self.linear(hidden_states))
        else:
            ret = self.linear(hidden_states)[..., None, :]
            ret = ret.repeat_interleave(self.n_block, dim=-2)
            return self.nonlinearity(ret.reshape(ret.size()[:-2] + (self.n_block * self.n_out,)))


class SoftmixOutputLayer(LMHead):

    def __init__(self,
                 config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_v = config.n_v
        self.vocab_size = config.vocab_size
        self.n_head = config.n_head
        self.sharing = config.sharing
        self.transform = config.transform
        self.conditional = config.conditional
        self.normalized = config.normalized
        self.l1_vocab_size = config.l1_vocab_size
        self.l2_vocab_size = config.l2_vocab_size
        self.language_heads = config.language_heads
        self.projection_nonlinearity = config.projection_nonlinearity

        # set up parameters
        self.ln_1 = nn.LayerNorm(self.n_embd, eps=config.layer_norm_epsilon)

        # transformation
        if self.transform == "projection":
            self.transform = ProjectionTransform(self.n_embd, self.n_v, self.n_head, nonlinearity=self.projection_nonlinearity)
        elif self.transform == "shared_projection":
            self.transform = ProjectionTransform(self.n_embd, self.n_v, self.n_head, shared=True, nonlinearity=self.projection_nonlinearity)
        elif self.transform == "cross_attention":
            assert config.attn_type == "softmix_multihead"
            self.transform = CrossAttention(config)
        else:
            raise ValueError(
                f"`config.transform` must be one of {{'projection', 'shared_projection', or 'cross_attention'}} (got `config.transform`: {config.transform})."
            )

        # output layer
        if self.sharing:
            self.output_layer = nn.Linear(self.n_v, self.vocab_size, bias=False)
        else:
            self.output_layer = BlockDiagonalProjection(self.n_v, self.vocab_size, self.n_head)

        if self.language_heads and self.n_head != 2:
            raise ValueError("Language heads flag only works when there are two heads.")

        # head mixer
        if self.conditional:
            self.head_classifier = nn.Linear(self.n_embd, self.n_head)
        else:
            self.head_classifier = ParameterLayer((self.n_head,))

        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ParameterLayer):
            module.value.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BlockDiagonalProjection):
            for matrix in module.l:
                matrix.data.normal_(mean=0.0, std=self.config.initializer_range)

    def init_weights(self):
        self.apply(self._init_weights)

    def forward(self,
                hidden_states=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        # reshape flatten
        output_batch_size = hidden_states.size()[:-2]
        hidden_states = hidden_states.reshape((-1,) + hidden_states.size()[-2:])
        attention_mask = attention_mask.reshape((-1,) + attention_mask.size()[-1:])
        encoder_hidden_states = encoder_hidden_states.reshape((-1,) + encoder_hidden_states.size()[-2:])
        encoder_attention_mask = encoder_attention_mask.reshape((-1,) + encoder_attention_mask.size()[-1:])

        # preprocessing
        attention_mask = attention_mask[:, None, None,:]
        encoder_attention_mask = encoder_attention_mask[:, None, None, :]

        # layer norm
        hidden_states = self.ln_1(hidden_states)

        # compute head mixture
        head_logits = self.head_classifier(hidden_states)[..., None]
        self.expose(head_logits, "head_logits")

        # compute head transform
        hidden_states = self.transform(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask)

        hidden_states = hidden_states.reshape(hidden_states.size()[:-1] + (self.n_head, self.n_v))

        # predict next token
        vocab_logits = self.output_layer(hidden_states)
        if self.language_heads:
            mask = vocab_logits.new_zeros(vocab_logits.size(), dtype=torch.bool)
            mask[:,:,0,self.l1_vocab_size:] = 1
            mask[:,:,1,4:self.l1_vocab_size] = 1
            vocab_logits = vocab_logits.masked_fill(mask, -float("inf"))
        self.expose(vocab_logits, "logits_by_head")

        # mix predictions bby heads
        if self.normalized:
            log_probs_by_head = torch.log_softmax(vocab_logits, dim=-1) # batch seq heads vocab
            log_weight_by_head = torch.log_softmax(head_logits, dim=-2)
            log_contrib_by_head = log_probs_by_head + log_weight_by_head # batch seq heads vocab
            logits = torch.logsumexp(log_contrib_by_head, dim=-2) # batch seq vocab # plus this should be normalized log probability

            if not self.training:
                self.expose(torch.softmax(head_logits.squeeze(-1), dim=-1), "mixture_probs") # batch seq head

        else:
            scaled_logits_by_head = head_logits + vocab_logits # batch seq heads vocab
            logits = torch.logsumexp(scaled_logits_by_head, dim=-2) # batch seq vocab # this is true logits, not normalized

            if not self.training:
                self.expose(torch.softmax(torch.logsumexp(scaled_logits_by_head, dim=-1), dim=-1), "mixture_probs") # batch seq head

        # reshape back
        logits = logits.reshape(output_batch_size + logits.size()[-2:])
        return logits





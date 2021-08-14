from torch.distributions import Categorical

from cslm.utils import ImmutableDict, max_gumbel, log_importance_weight
from collections import defaultdict

import torch

def grow_prefixes(t=None,
                  decoder_input_ids=None,
                  cum_log_probs=None,
                  eos_ids=None,
                  final_log_probs=None,
                  next_log_probs=None,
                  last_index=None):
    """
    Returns new decoder_input_ids, decoder_attention_mask, cum_log_probs.

    :param decoder_input_ids:
    :param decoder_attention_mask:
    :param cum_log_probs:
    :param eos_ids:
    :param pad_id:
    :param final_log_probs:
    :param final_log_probs:
    :param next_log_probs:
    :return:
    """

    cum_log_probs = cum_log_probs.detach().clone()

    # reshaping
    prefixes_view = decoder_input_ids.view(-1, decoder_input_ids.size(-1))

    samples = Categorical(logits=next_log_probs).sample()
    sample_log_probs = torch.gather(next_log_probs, -1, samples[:, :, None]).squeeze(-1)
    cum_log_probs += sample_log_probs
    is_eos = samples.new_full(samples.size(), 0, dtype=torch.bool)
    for eos_id in eos_ids:
        is_eos =  is_eos | (samples == eos_id)
    mask = (is_eos & (last_index > t)) | (last_index == t)
    final_log_probs = final_log_probs.masked_fill(mask, 0) + cum_log_probs.masked_fill(~mask, 0)
    last_index = last_index.masked_fill(mask, 0) + last_index.new_full(last_index.size(), t).masked_fill(~mask, 0)

    new_decoder_input_ids = torch.cat((decoder_input_ids, samples.unsqueeze(-1)),dim=-1)
    new_decoder_attention_mask = new_decoder_input_ids.new_full(new_decoder_input_ids.size(), 1)
    return {
        "decoder_input_ids": new_decoder_input_ids,
        "decoder_attention_mask": new_decoder_attention_mask,
        "cum_log_probs": cum_log_probs,
        "final_log_probs": final_log_probs,
        "last_index":last_index
    }


@torch.no_grad()
def sample(model=None,
           input_ids=None,
           attention_mask=None,
           decoder_input_ids=None,
           decoder_attention_mask=None,
           max_length=None,
           num_return_sequences=None,
           bos_id=None,
           eos_ids=tuple(),
           vocab_size=None,
           pad_id=None,
           ):
    # assert beam_size
    batch_size = input_ids.size(0)
    device = input_ids.device

    # expand dimensions of input ids
    input_ids = input_ids[:, None, ...].expand(input_ids.size(0), num_return_sequences, *input_ids.size()[1:]) # batch bin beam seq
    attention_mask = attention_mask[:, None, ...].expand(attention_mask.size(0), num_return_sequences, *attention_mask.size()[1:]) # batch bin beam seq

    # initialize decoder with [BOS]
    if decoder_input_ids is None:
        decoder_input_ids = torch.full((batch_size, num_return_sequences, 1), bos_id, dtype=torch.long, device=device)
    if decoder_attention_mask is None:
        decoder_attention_mask = torch.full((batch_size, num_return_sequences, 1), 1, dtype=torch.long, device=device)

    # dynamic book keeping
    cum_log_probs = torch.full((batch_size, num_return_sequences), 0.0, dtype=torch.float32, device=device)
    final_log_probs = torch.full((batch_size, num_return_sequences), -float("inf"), dtype=torch.float32, device=device)
    last_index = torch.full((batch_size, num_return_sequences), max_length-1, dtype=torch.long, device=device)


    # extract intermediate outputs necessary for computation
    model.add_exposure_pattern("encoder_last_layer")
    for t in range(max_length - 1):
        decoder_last_layer = model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        encoder_last_layer = dict(model.named_exposed_tensors())["base_model.encoder_last_layer"]
        model.release_exposed_tensors()
        logits = model.lm_head(hidden_states=decoder_last_layer,
                               attention_mask=decoder_attention_mask,
                               encoder_hidden_states=encoder_last_layer,
                               encoder_attention_mask=attention_mask)[..., -1, :]
        next_word_log_probs = torch.log_softmax(logits, dim=-1)

        outputs = grow_prefixes(t=t+1,
                                decoder_input_ids=decoder_input_ids,
                                cum_log_probs=cum_log_probs,
                                eos_ids=eos_ids,
                                final_log_probs=final_log_probs,
                                next_log_probs=next_word_log_probs,
                                last_index=last_index)
        decoder_input_ids = outputs["decoder_input_ids"]
        decoder_attention_mask = outputs["decoder_attention_mask"]
        cum_log_probs = outputs["cum_log_probs"]
        final_log_probs = outputs["final_log_probs"]
        last_index = outputs["last_index"]
    decoder_attention_mask = (torch.arange(max_length, device=device)[None, None,:].expand_as(decoder_attention_mask) <= last_index.unsqueeze(-1)).to(torch.long)
    return {
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "log_probs": torch.maximum(final_log_probs, cum_log_probs).tolist()
    }

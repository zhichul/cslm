from cslm.utils import ImmutableDict
from collections import defaultdict

import torch
import heapq


class BeamCell:

    def __init__(self, size):
        self.size = size
        self.beams = []
        self.worst_score = None
        self.frozen = False

    def add(self, score, ids, state):
        if self.frozen:
            raise ValueError("Cannot add to a frozen beam cell.")
        if len(self.beams) == self.size:
            if self.worst_score is None or score > self.worst_score:
                heapq.heappushpop(self.beams, (score, ids, state))
        else:
            heapq.heappush(self.beams, (score, ids, state))
        self.worst_score = self.beams[0][0]

    def freeze(self):
        self.frozen = True
        heap = self.beams
        beams = []
        for i in range(len(heap)):
            score, ids, state = heapq.heappop(heap)
            beams.append((score, tuple(ids), ImmutableDict({k: v for k, v in state.items()})))
        self.beams = tuple(beams)

    def __iter__(self):
        if not self.frozen:
            self.freeze()
        return iter(self.beams)

    def __reversed__(self):
        if not self.frozen:
            self.freeze()
        return reversed(self.beams)

    def __len__(self):
        return len(self.beams)


def grow_beams(cells=None,
               beam_ids=None,
               tokens=None,
               num_beams=None,
               num_bins=None,
               decoder_input_ids=None,
               decoder_attention_mask=None,
               state=None,
               cum_log_probs=None,
               extension_state=None,
               extension_bin_assignment=None,
               eos_ids=None,
               pad_id=None):
    """
    Updates the cells in-place, and returns new decoder_input_ids, decoder_attention_mask, cum_log_probs, and state.

    :param cells:
    :param beam_ids:
    :param tokens:
    :param num_beams:
    :param num_bins:
    :param decoder_input_ids:
    :param decoder_attention_mask:
    :param state:
    :param cum_log_probs:
    :param extension_state:
    :param extension_bin_assignment:
    :param eos_ids:
    :param pad_id:
    :return:
    """
    # reshaping
    prefixes_view = decoder_input_ids.view(-1, decoder_input_ids.size(-1))

    # tmp results
    next_tokens_by_bin = []
    prefixes_by_bin = []
    cum_log_probs_by_bin = []
    state_by_bin = defaultdict(list)
    for b in range(num_bins):
        mask = extension_bin_assignment == b

        # select bin
        cum_log_probs_of_bin = torch.masked_select(cum_log_probs, mask)
        beam_ids_of_bin = torch.masked_select(beam_ids, mask)
        tokens_of_bin = torch.masked_select(tokens, mask)
        extension_state_of_bin = {k: torch.masked_select(v, mask) for k, v in extension_state.items()}

        # select top k
        actual_retained_extensions = min(cum_log_probs_of_bin.numel(), (
                num_bins + 1) * num_beams)  # at most num_bins * num_beams extensions have EOS, and will thus not continue to be searched
        topk_ret = torch.topk(cum_log_probs_of_bin, actual_retained_extensions, dim=-1)
        topk_beam_ids_of_bin = beam_ids_of_bin[topk_ret.indices]
        topk_cum_log_probs_of_bin = cum_log_probs_of_bin[topk_ret.indices]
        topk_tokens_of_bin = tokens_of_bin[topk_ret.indices]
        topk_extension_state_of_bin = {k: v[topk_ret.indices] for k, v in extension_state_of_bin.items()}

        next_tokens_of_bin = []
        prefixes_of_bin = []
        new_cum_log_probs_of_bin = []
        state_of_bin = defaultdict(list)
        # pull of extensions with EOS and try to fill the bin with the remaining open extensions
        for i in range(actual_retained_extensions):
            # if beam has been filled, stop
            if len(next_tokens_of_bin) == num_beams:
                break
            candidate_next_token = topk_tokens_of_bin[i]
            candidate_beam_id = topk_beam_ids_of_bin[i]
            candidate_cum_log_prob = topk_cum_log_probs_of_bin[i]
            candidate_prefix = prefixes_view[candidate_beam_id.item()]
            if candidate_next_token.item() in eos_ids:
                cells[b].add(score=candidate_cum_log_prob.item(),
                             ids=candidate_prefix.tolist() + [candidate_next_token.item()],
                             state={k: v[i].tolist() for k, v in topk_extension_state_of_bin.items()})
            else:
                next_tokens_of_bin.append(candidate_next_token.tolist())
                prefixes_of_bin.append(candidate_prefix.tolist())
                new_cum_log_probs_of_bin.append(candidate_cum_log_prob.tolist())
                for k, v in topk_extension_state_of_bin.items():
                    state_of_bin[k].append(v[i].tolist())
        # fill the rest of the bin with padding
        for _ in range(num_beams - len(next_tokens_of_bin)):
            j = 0
            candidate_next_token = decoder_input_ids.new_tensor(pad_id)
            candidate_cum_log_prob = cum_log_probs.new_tensor(-1e9)
            candidate_prefix = prefixes_view[b * num_beams]

            next_tokens_of_bin.append(candidate_next_token.tolist())
            prefixes_of_bin.append(candidate_prefix.tolist())
            new_cum_log_probs_of_bin.append(candidate_cum_log_prob.tolist())
            for k, v in state.items():
                state_of_bin[k].append(v[0, b, 0, 0].tolist())

        next_tokens_by_bin.append(next_tokens_of_bin)
        prefixes_by_bin.append(prefixes_of_bin)
        cum_log_probs_by_bin.append(new_cum_log_probs_of_bin)
        for k, v in state_of_bin.items():
            state_by_bin[k].append(v)
    # pack
    packed_next_tokens = torch.tensor(next_tokens_by_bin)[..., None]  # num_bins x num_beams x 1
    packed_prefixes = torch.tensor(prefixes_by_bin)  # num_bins x num_beams x prefix_length
    decoder_input_ids = torch.cat((packed_prefixes, packed_next_tokens), dim=-1)[
        None, ...]  # batch_size x num_bins x num_beams x new_prefix_length
    decoder_attention_mask = decoder_input_ids.new_ones(decoder_input_ids.size())
    cum_log_probs = torch.tensor(cum_log_probs_by_bin)[None, ..., None]
    state = {k: torch.tensor(v)[None, ..., None] for k, v in state_by_bin.items()}
    return {
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "cum_log_probs": cum_log_probs,
        "state": state
    }


@torch.no_grad()
def beam_search(model=None,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                max_length=None,
                num_beams=None,
                num_return_sequences=None,
                num_bins=None,
                initial_state=None,
                fn_update_state=None,
                fn_assign_bin=None,
                bos_id=None,
                eos_ids=tuple(),
                vocab_size=None,
                pad_id=None,
                ):
    # assert beam_size
    batch_size = input_ids.size(0)
    assert batch_size == 1

    # initialize decoder with [BOS]
    if decoder_input_ids is None:
        decoder_input_ids = torch.full((batch_size, num_bins, num_beams, 1), bos_id, dtype=torch.long)
    if decoder_attention_mask is None:
        decoder_attention_mask = torch.full((batch_size, num_bins, num_beams, 1), 1, dtype=torch.long)

    # constant book keeping
    tokens = torch.arange(0, vocab_size, step=1, dtype=torch.long)[None, None, None, ...]
    tokens = tokens.expand(batch_size, num_bins, num_beams, vocab_size)
    beam_ids = torch.arange(0, batch_size * num_bins * num_beams)[..., None]
    beam_ids = beam_ids.expand(batch_size * num_bins * num_beams, vocab_size).view(batch_size, num_bins, num_beams,
                                                                                   vocab_size)

    # dynamic book keeping
    state = initial_state

    cum_log_probs = torch.full((batch_size, num_bins, num_beams, 1), -1e9, dtype=torch.float32)
    cum_log_probs[0, 0, 0, 0] = 0.0

    cells = [BeamCell(num_return_sequences) for _ in range(num_bins)]

    for t in range(max_length - 1):
        decoder_last_layer = model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = model.lm_head(decoder_last_layer)[..., -1, :]
        next_word_log_probs = torch.log_softmax(logits, dim=-1)

        # expansion (broadcasting will happen here)
        cum_log_probs = cum_log_probs + next_word_log_probs

        # expanded states and bin assignments
        extension_state = fn_update_state(state, tokens)
        extension_bin_assignment = fn_assign_bin(extension_state)

        outputs = grow_beams(cells=cells,
                             beam_ids=beam_ids,
                             tokens=tokens,
                             num_beams=num_beams,
                             num_bins=num_bins,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             state=state,
                             cum_log_probs=cum_log_probs,
                             extension_state=extension_state,
                             extension_bin_assignment=extension_bin_assignment,
                             eos_ids=eos_ids,
                             pad_id=pad_id)

        decoder_input_ids = outputs["decoder_input_ids"]
        decoder_attention_mask = outputs["decoder_attention_mask"]
        cum_log_probs = outputs["cum_log_probs"]
        state = outputs["state"]
    for b, cell in enumerate(cells):
        if len(cell) < num_return_sequences:
            for i in range(num_return_sequences - len(cell)):
                cell.add(cum_log_probs[0, b, i, 0].item(), decoder_input_ids[0, b, i, :].tolist(), {k: v[0, b, i, 0].tolist() for k,v in state.items()})
    for cell in cells:
        cell.freeze()
    return cells

import torch

NUM_BINS = 6

def assign_bin_factory():
    def assign_bin(prefix_state):
        switch_count = prefix_state["switch_count"]
        null_switch = switch_count == 0
        bin_1 = switch_count == 1
        bin_2 = switch_count == 2
        bin_3 = switch_count == 3
        bin_4 = switch_count == 4
        bin_5 = switch_count == 5
        return torch.where(null_switch, 0,
               torch.where(bin_1, 1,
               torch.where(bin_2, 2,
               torch.where(bin_3, 3,
               torch.where(bin_4, 4,
               # bin_5 is the else case
               5)))))
    return assign_bin


def update_state_factory(eos_ids, l1_vocab_size=304):
    def update_state(prefix_state, token):
        next_is_l2 = (token >= l1_vocab_size)
        next_is_eos = token.new_zeros(token.size(), dtype=torch.bool)
        prev_is_bos = prefix_state["last_token"] == -100
        for eos_id in eos_ids:
            next_is_eos = next_is_eos | (token == eos_id)
        state = {
            "l2_count": torch.where(next_is_eos, prefix_state["l2_count"],
                                    prefix_state["l2_count"] + next_is_l2),
            "tok_count":  torch.where(next_is_eos, prefix_state["tok_count"],
                                      prefix_state["tok_count"] + next_is_l2.new_ones(next_is_l2.size(), device=next_is_l2.device)),
            "fence_count": torch.where(next_is_eos | prev_is_bos, prefix_state["fence_count"],
                                      prefix_state["fence_count"] + next_is_l2.new_ones(next_is_l2.size(), device=next_is_l2.device)),
            "switch_count": torch.where(next_is_eos | prev_is_bos, prefix_state["switch_count"],
                                    prefix_state["switch_count"] + (next_is_l2 != prefix_state["last_token_lang"]).to(torch.long)),
            "last_token": token,
            "last_token_lang": next_is_l2.to(torch.long),
        }
        return state
    return update_state


def initial_state_factory():
    def initial_state(batch_size, num_bins, num_beams, device="cpu"):
        return {
            "l2_count": torch.zeros((batch_size, num_bins, num_beams, 1), dtype=torch.long, device=device),
            "tok_count": torch.zeros((batch_size, num_bins, num_beams, 1), dtype=torch.long, device=device),
            "last_token": torch.full((batch_size, num_bins, num_beams, 1), -100, dtype=torch.long, device=device),
            "last_token_lang": torch.full((batch_size, num_bins, num_beams, 1), -1, dtype=torch.long, device=device),
            "fence_count": torch.full((batch_size, num_bins, num_beams, 1), 0, dtype=torch.long, device=device),
            "switch_count": torch.full((batch_size, num_bins, num_beams, 1), 0, dtype=torch.long, device=device),
        }
    return initial_state
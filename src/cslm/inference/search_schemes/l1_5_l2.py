import torch

NUM_BINS = 7

def assign_bin_factory():
    def assign_bin(prefix_state):
        l2_percentage = prefix_state["l2_count"] / prefix_state["tok_count"]
        full_l2 = l2_percentage == 1.0
        full_l1 = l2_percentage == 0.0
        bin_1 = (0.0 < l2_percentage) & (l2_percentage <= 1/5)
        bin_2 = (1/5 < l2_percentage) & (l2_percentage <= 2/5)
        bin_3 = (2/5 < l2_percentage) & (l2_percentage < 3/5)
        bin_4 = (3/5 <= l2_percentage) & (l2_percentage < 4/5)
        bin_5 = (4/5 <= l2_percentage) & (l2_percentage < 1.0)
        return torch.where(full_l1, 0,
               torch.where(bin_1, 1,
               torch.where(bin_2, 2,
               torch.where(bin_3, 3,
               torch.where(bin_4, 4,
               torch.where(bin_5, 5,
               # full_l2 is the else case
               6))))))
    return assign_bin


def update_state_factory(eos_ids, l1_vocab_size=304):
    def update_state(prefix_state, token):
        next_is_l2 = (token >= l1_vocab_size)
        next_is_eos = token.new_zeros(token.size(), dtype=torch.bool)
        for eos_id in eos_ids:
            next_is_eos = next_is_eos | (token == eos_id)
        state = {
            "l2_count": torch.where(next_is_eos, prefix_state["l2_count"],
                                    prefix_state["l2_count"] + next_is_l2),
            "tok_count":  torch.where(next_is_eos, prefix_state["tok_count"],
                                      prefix_state["tok_count"] + next_is_l2.new_ones(next_is_l2.size(), device=next_is_l2.device)),
            "last_token": token
        }
        return state
    return update_state


def initial_state_factory():
    def initial_state(batch_size, num_bins, num_beams, device="cpu"):
        return {
            "l2_count": torch.zeros((batch_size, num_bins, num_beams, 1), dtype=torch.long, device=device),
            "tok_count": torch.zeros((batch_size, num_bins, num_beams, 1), dtype=torch.long, device=device),
            "last_token": torch.full((batch_size, num_bins, num_beams, 1), -100, dtype=torch.long, device=device),
        }
    return initial_state
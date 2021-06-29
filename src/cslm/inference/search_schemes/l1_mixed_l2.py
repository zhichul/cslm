import torch


def assign_bin_factory():
    def assign_bin(prefix_state):
        full_l2 = prefix_state["l2_count"] == prefix_state["tok_count"]
        full_l1 = prefix_state["l2_count"] == 0
        return torch.where(full_l1, 0, torch.where(full_l2, 2, 1))
    return assign_bin


def update_state_factory(eos_ids):
    def update_state(prefix_state, token):
        next_is_l2 = (token >= 304)
        next_is_eos = token.new_zeros(token.size(), dtype=torch.bool)
        for eos_id in eos_ids:
            next_is_eos = next_is_eos | (token == eos_id)
        state = {
            "l2_count": torch.where(next_is_eos, prefix_state["l2_count"],
                                    prefix_state["l2_count"] + next_is_l2),
            "tok_count":  torch.where(next_is_eos, prefix_state["tok_count"],
                                      prefix_state["tok_count"] + next_is_l2.new_ones(next_is_l2.size()))
        }
        return state
    return update_state


def initial_state(batch_size=None, num_bins=None, num_beams=None):
    return {
        "l2_count": torch.zeros((batch_size, num_bins, num_beams, 1), dtype=torch.long),
        "tok_count": torch.zeros((batch_size, num_bins, num_beams, 1), dtype=torch.long)
    }
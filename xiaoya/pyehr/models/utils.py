import torch


def generate_mask(seq_lens):
    """Generates a mask for the sequence.

    Args:
        seq_lens: [batch size]
        (max_len: int)

    Returns:
        mask: [batch size, max_len]
    """
    max_len = torch.max(seq_lens).to(seq_lens.device)
    mask = torch.arange(max_len).expand(len(seq_lens), int(max_len)).to(seq_lens.device)
    mask = mask < seq_lens.unsqueeze(1)
    return mask

def get_last_visit(hidden_states, mask):
    """Gets the last visit from the sequence model.

    Args:
        hidden_states: [batch size, seq len, hidden_size]
        mask: [batch size, seq len]

    Returns:
        last_visit: [batch size, hidden_size]
    """
    if mask is None:
        return hidden_states[:, -1, :]
    else:
        mask = mask.long()
        last_visit = torch.sum(mask, 1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state


import torch


@torch.no_grad()
def build_turn_loss_weights(action_mask, rollouts):
    weights = torch.zeros_like(action_mask, dtype=torch.float32)
    for row, rollout in enumerate(rollouts):
        turns = rollout.get("turns", [])
        coverage = torch.zeros_like(action_mask[row], dtype=torch.bool)
        end_previous = 0
        for turn in turns:
            start, end = (turn["token_start"], turn["token_end"])
            if not 0 <= end_previous <= start < end <= action_mask.shape[1]:
                break
            coverage[start:end] = True
            weights[row, start:end] = 1.0 / (len(turns) * (end - start))
            end_previous = end
        else:
            if torch.equal(coverage, action_mask[row].bool()):
                continue
        weights[row].zero_()
    return weights

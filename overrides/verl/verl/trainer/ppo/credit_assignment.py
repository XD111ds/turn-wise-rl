import math
from collections import Counter, defaultdict
import torch


def credit_alpha(
    optimizer_steps, total_optimizer_steps, alpha_max=1.0, warmup_ratio=0.1
):
    if not math.isfinite(alpha_max) or alpha_max < 0 or (not 0 <= warmup_ratio <= 1):
        raise ValueError(
            "credit alpha must be finite/nonnegative; warmup_ratio must be in [0, 1]"
        )
    warmup = math.ceil(total_optimizer_steps * warmup_ratio)
    return alpha_max if warmup == 0 else alpha_max * min(1.0, optimizer_steps / warmup)


@torch.no_grad()
def terminal_normalized_turn_advantages(scores, prefixes, alpha, epsilon=1e-06):
    std = scores.std() if scores.numel() > 1 else scores.new_zeros(())
    if std == 0:
        return prefixes[:, 1:] * 0
    global_adv = (scores - scores.mean()) / (std + epsilon)
    earned = prefixes[:, :-1] - prefixes[:, :1]
    return global_adv[:, None] - alpha * (earned - earned.mean(dim=0)) / (std + epsilon)


def _check_record(record, mask, score):
    if not record.get("valid", False):
        return record.get("reason", "invalid_reward")
    turns = record["rollout"]["turns"]
    ids = record["turn_ids"]
    if not turns or ids != [r["turn_id"] for r in turns] or len(set(ids)) != len(ids):
        return "turn_identity_mismatch"
    coverage = torch.zeros_like(mask)
    previous_end = 0
    for turn in turns:
        start, end = (turn["token_start"], turn["token_end"])
        if not 0 <= previous_end <= start < end <= mask.numel():
            return "token_range_mismatch"
        coverage[start:end] = 1
        previous_end = end
    if not torch.equal(coverage.bool(), mask.bool()):
        return "assistant_mask_mismatch"
    prefixes = record["prefix_scores"]
    if len(prefixes) != len(turns) + 1 or not all((math.isfinite(x) for x in prefixes)):
        return "invalid_prefix_scores"
    if not math.isclose(prefixes[-1], float(score), rel_tol=1e-05, abs_tol=1e-05):
        return "terminal_reward_mismatch"
    return None


@torch.no_grad()
def assign_turn_credit(
    token_rewards,
    mask,
    group_ids,
    records,
    global_advantages,
    *,
    alpha,
    enabled=True,
    epsilon=1e-06,
    clip=5.0,
    group_size=4
):
    if (
        epsilon <= 0
        or not math.isfinite(epsilon)
        or clip <= 0
        or (not math.isfinite(clip))
    ):
        raise ValueError("credit epsilon and clip must be finite and positive")
    scores = token_rewards.sum(-1)
    advantages = torch.zeros_like(global_advantages)
    groups = defaultdict(list)
    for i, uid in enumerate(group_ids):
        groups[str(uid)].append(i)
    reasons = Counter()
    diagnostics = [None] * len(records)
    stds, raw_values, earned_values = ([], [], [])
    clipped = total_tokens = valid_groups = used_groups = 0
    for uid, indices in groups.items():
        group_records = [records[i] for i in indices]
        problem = "group_size_mismatch" if len(indices) != group_size else None
        for i, rec in zip(indices, group_records):
            problem = problem or _check_record(rec, mask[i], scores[i])
        group_scores = scores[indices]
        if not torch.isfinite(group_scores).all():
            problem = "nonfinite_reward"
        complete = all((r.get("complete", False) for r in group_records))
        if not problem:
            keys = [r["evaluation_key"] for r in group_records]
            for key in keys[1:]:
                for field in (
                    "question_id",
                    "expected_turn_ids",
                    "decision_times",
                    "previous_replies",
                ):
                    if key[field] != keys[0][field]:
                        problem = "group_identity_mismatch"
            if complete and (
                any((key != keys[0] for key in keys[1:]))
                or any(
                    (
                        r["turn_ids"] != group_records[0]["turn_ids"]
                        for r in group_records[1:]
                    )
                )
                or any(
                    (
                        not math.isclose(
                            r["prefix_scores"][0],
                            group_records[0]["prefix_scores"][0],
                            abs_tol=1e-08,
                        )
                        for r in group_records[1:]
                    )
                )
            ):
                problem = "prefix_alignment_mismatch"
        if problem:
            status = "invalid/" + problem
        else:
            valid_groups += 1
            std = group_scores.std() if len(indices) > 1 else group_scores.new_zeros(())
            stds.append(float(std))
            if std == 0:
                status = "zero_variance"
            elif not enabled:
                status = "baseline"
                advantages[indices] = global_advantages[indices]
            elif not complete:
                status = "fallback/truncated"
                advantages[indices] = global_advantages[indices]
            else:
                status = "turn_credit"
                prefixes = scores.new_tensor(
                    [r["prefix_scores"] for r in group_records]
                )
                per_turn = terminal_normalized_turn_advantages(
                    group_scores, prefixes, alpha, epsilon
                )
                earned_values.extend(
                    (prefixes[:, :-1] - prefixes[:, :1]).flatten().tolist()
                )
                for row, i in enumerate(indices):
                    for t, turn in enumerate(records[i]["rollout"]["turns"]):
                        advantages[i, turn["token_start"] : turn["token_end"]] = (
                            per_turn[row, t]
                        )
                used_groups += 1
            active = advantages[indices][mask[indices].bool()]
            raw_values.extend(active.tolist())
            clipped += int((active.abs() > clip).sum()) if enabled else 0
            total_tokens += active.numel()
        reasons[status] += 1
        for i in indices:
            diagnostics[i] = {
                "group_id": uid,
                "request_id": records[i].get("rollout", {}).get("request_id"),
                "status": status,
                "alpha": alpha if enabled and status == "turn_credit" else 0.0,
                "global_advantage": (
                    float(global_advantages[i][mask[i].bool()][0])
                    if mask[i].any()
                    else 0.0
                ),
                "turn_advantages_unclipped": [
                    float(advantages[i, t["token_start"]])
                    for t in records[i].get("rollout", {}).get("turns", [])
                    if 0 <= t["token_start"] < advantages.shape[1]
                ],
            }
    if enabled:
        advantages.clamp_(-clip, clip)
    advantages *= mask
    ngroups = max(len(groups), 1)
    metrics = {
        "credit/alpha": alpha if enabled else 0.0,
        "credit/valid_group_fraction": valid_groups / ngroups,
        "credit/used_group_fraction": used_groups / ngroups,
        "credit/invalid_group_fraction": (len(groups) - valid_groups) / ngroups,
        "credit/clip_fraction": clipped / max(total_tokens, 1),
        "credit/terminal_std_mean": sum(stds) / max(len(stds), 1),
        "credit/small_nonzero_std_fraction": sum((0 < s < epsilon for s in stds))
        / ngroups,
    }
    metrics.update(
        {
            "credit/" + key + "_fraction": value / ngroups
            for key, value in reasons.items()
        }
    )
    for name, values in (
        ("advantage_unclipped", raw_values),
        ("earned", earned_values),
    ):
        metrics["credit/" + name + "_mean"] = sum(values) / max(len(values), 1)
        metrics["credit/" + name + "_min"] = min(values, default=0.0)
        metrics["credit/" + name + "_max"] = max(values, default=0.0)
    return (advantages, metrics, diagnostics, valid_groups > 0)

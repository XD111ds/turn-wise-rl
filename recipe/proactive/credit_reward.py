import math

SCORER_VERSION = "history-validated-judge-v2-20261008"


def _invalid(rollout, reason):
    return {
        "score": 0.0,
        "reward_valid": 0.0,
        "reward_pauc": 0.0,
        "reward_repetition": 0.0,
        "reward_outspan": 0.0,
        "reward_common_prefix": 0.0,
        "judge_calls": 0,
        "silent_fraction": 0.0,
        "reply_count": 0,
        "credit_records": {"valid": False, "reason": reason, "rollout": rollout},
    }


def score_trajectory(ground_truth, extra_info, api):
    from recipe.proactive.judge_client import start_trace, finish_trace

    rollout = (extra_info or {}).get("credit_rollout") or {}
    ground_truth = dict(ground_truth)
    if "previous_replies" in rollout:
        for key in ("previous_replies", "history_scope", "history_message_indices"):
            if key in rollout:
                ground_truth[key] = rollout[key]
    previous = list(
        ground_truth.get(
            "previous_replies", ground_truth.get("prev_assistant_replies", [])
        )
    )
    ground_truth["previous_replies"] = previous
    tokens = start_trace(
        {
            "question_id": ground_truth.get("question_id"),
            "answers": ground_truth.get("answer"),
            "previous_replies": previous,
            "history_scope": ground_truth.get("history_scope", "legacy_ground_truth"),
        }
    )
    try:
        result = _score_trajectory(ground_truth, extra_info, api)
    finally:
        events = finish_trace(tokens)
    turns = rollout.get("turns", [])
    result["reply_count"] = sum(
        (t.get("text", "") not in ("", api.NO_REPLY_STR) for t in turns)
    )
    result["silent_fraction"] = sum(
        (t.get("text") == api.NO_REPLY_STR for t in turns)
    ) / max(len(turns), 1)
    result.update(
        judge_requests=sum((e.get("physical_request", False) for e in events)),
        judge_retries=sum(
            (e.get("physical_request", False) and e["attempt"] > 1 for e in events)
        ),
        judge_failed_attempts=sum((bool(e.get("error_type")) for e in events)),
        judge_cache_hits=sum((e["cache_hit"] for e in events)),
        judge_seconds=sum((e["elapsed_seconds"] for e in events)),
        history_reply_count=len(previous),
        history_metadata_unmatched_count=ground_truth.get(
            "history_metadata_unmatched_count", 0
        ),
    )
    result["judge_repair_calls"] = result["credit_records"].get("judge_repair_calls", 0)
    result["judge_recovered_items"] = result["credit_records"].get(
        "judge_recovered_items", 0
    )
    result["credit_records"].update(
        scorer_version=SCORER_VERSION,
        previous_replies=previous,
        history_scope=ground_truth.get("history_scope", "legacy_ground_truth"),
        history_message_indices=ground_truth.get("history_message_indices", []),
        judge_attempts=events,
        reward_protocol={
            "repetition_denominator": "replies",
            "planned_turn_count": len(rollout.get("expected_turn_ids", [])),
            "weights": {
                "pauc": api.PAUC_WEIGHT,
                "repetition": api.REPETITION_WEIGHT,
                "outspan": api.OUTSPAN_WEIGHT,
                "common_prefix": api.COMMON_PREFIX_WEIGHT,
            },
        },
    )
    return result


def _score_trajectory(ground_truth, extra_info, api):
    rollout = (extra_info or {}).get("credit_rollout")
    if not rollout:
        return _invalid({}, "missing_rollout_identity")
    turns = rollout.get("turns", [])
    expected = rollout.get("expected_turn_ids", [])
    times = list(ground_truth["model_pred_timestamps"])
    ids = [turn["turn_id"] for turn in turns]
    if (
        not turns
        or len(times) != len(expected)
        or len(set(expected)) != len(expected)
        or (ids != expected[: len(ids)])
        or (len(ids) > len(expected))
        or (rollout.get("question_id") != ground_truth["question_id"])
    ):
        return _invalid(rollout, "turn_identity_mismatch")
    if not all((math.isfinite(float(t)) for t in times)) or times != sorted(times):
        return _invalid(rollout, "invalid_decision_times")
    if any((not turn["text"] for turn in turns)):
        return _invalid(rollout, "empty_assistant_reply")
    complete = (
        ids == expected
        and rollout.get("end_reason") == "input_exhausted"
        and (not rollout.get("truncated", False))
    )
    windows = []
    for window_id, answer in enumerate(ground_truth["answer"]):
        a, b = map(float, answer["reply_timespan"])
        if not (math.isfinite(a) and math.isfinite(b) and (b > a)):
            return _invalid(rollout, "invalid_reply_window")
        if a <= times[len(turns) - 1]:
            windows.append(
                {
                    "window_id": window_id,
                    "start": a,
                    "end": b,
                    "question": answer["question"],
                    "content": answer["content"],
                }
            )
    previous = list(ground_truth["previous_replies"])
    repetition_enabled = api.REPETITION_WEIGHT
    seen_all = previous.copy()
    seen_window = [[] for _ in windows]
    records = []
    judge_calls = 0
    failed_jobs = []

    def judge(fn, args, target, kind):
        nonlocal judge_calls
        _, value = fn(*args)
        judge_calls += 1
        valid = (
            isinstance(value, bool)
            if kind == "repetition"
            else value is not None
            and math.isfinite(float(value))
            and (0 <= value <= api.MAX_SCORES - 1)
        )
        target["value"] = value if valid else None
        if not valid:
            failed_jobs.append((fn, args, target, kind))

    for index, turn in enumerate(turns):
        time = float(times[index])
        text = turn["text"]
        rec = {**turn, "decision_time": time, "similarity": [], "repetition": []}
        if text != api.NO_REPLY_STR:
            if api.REPETITION_IN_ALL_SPANS and repetition_enabled:
                if text in seen_all:
                    scored = {"window_id": None, "value": True}
                else:
                    scored = {"window_id": None}
                    judge(
                        api.calculate_is_repetition,
                        (time, " ".join(seen_all), text),
                        scored,
                        "repetition",
                    )
                rec["repetition"].append(scored)
                if text not in seen_all:
                    seen_all.append(text)
            for w, window in enumerate(windows):
                if not window["start"] <= time <= window["end"]:
                    continue
                seen = seen_window[w]
                if text in seen:
                    if not api.REPETITION_IN_ALL_SPANS and repetition_enabled:
                        rec["repetition"].append(
                            {"window_id": window["window_id"], "value": True}
                        )
                    continue
                seen.append(text)
                if api.PAUC_WEIGHT:
                    pred = " ".join(seen) if api.PAUC_ADD_PREV_TURNS else text
                    scored = {"window_id": window["window_id"]}
                    judge(
                        api.calculate_similarity_score,
                        (time, window["question"], pred, window["content"]),
                        scored,
                        "similarity",
                    )
                    rec["similarity"].append(scored)
                if (
                    not api.REPETITION_IN_ALL_SPANS
                    and repetition_enabled
                    and (len(seen) > 1)
                ):
                    scored = {"window_id": window["window_id"]}
                    judge(
                        api.calculate_is_repetition,
                        (time, " ".join(seen[:-1]), text),
                        scored,
                        "repetition",
                    )
                    rec["repetition"].append(scored)
        records.append(rec)
    initial_failed_count = len(failed_jobs)
    repair_calls = 0
    for _ in range(getattr(api, "TRAJECTORY_RETRIES", 1)):
        pending, failed_jobs = (failed_jobs, [])
        for job in pending:
            judge(*job)
            repair_calls += 1
        if not failed_jobs:
            break
    repair = {
        "judge_repair_calls": repair_calls,
        "judge_recovered_items": initial_failed_count - len(failed_jobs),
    }
    if failed_jobs:
        failure = "judge_" + failed_jobs[0][3] + "_failed"
        result = _invalid(rollout, failure)
        result["judge_calls"] = judge_calls
        result["credit_records"].update(
            score_records=records, windows=windows, judge_calls=judge_calls
        )
        result["credit_records"].update(repair)
        return result
    prefixes, components = replay_prefixes(records, windows, previous, api)
    result = {
        "score": prefixes[-1],
        "reward_valid": 1.0,
        "judge_calls": judge_calls,
        "silent_fraction": sum((t["text"] == api.NO_REPLY_STR for t in turns))
        / len(turns),
        "reply_count": sum((t["text"] != api.NO_REPLY_STR for t in turns)),
        "credit_records": {
            "valid": True,
            "reason": "",
            "complete": complete,
            "rollout": rollout,
            "turn_ids": ids,
            "decision_times": times[: len(turns)],
            "windows": windows,
            "previous_replies": previous,
            "prefix_scores": prefixes,
            **repair,
            "score_records": records,
            "judge_calls": judge_calls,
            "evaluation_key": {
                "question_id": ground_truth["question_id"],
                "expected_turn_ids": expected,
                "decision_times": times,
                "windows": windows,
                "previous_replies": previous,
            },
        },
    }
    result.update(components)
    return result


def replay_prefixes(records, windows, previous, api):
    count = len(records)
    max_score = api.MAX_SCORES - 1
    start_score = 0.5 / max_score
    quality = {w["window_id"]: start_score for w in windows}
    window_by_id = {w["window_id"]: w for w in windows}
    auc = {w["window_id"]: start_score for w in windows}
    outside = [
        not any((w["start"] <= r["decision_time"] <= w["end"] for w in windows))
        for r in records
    ]
    outside_count = sum(outside)
    repetition_sum = repetition_count = out_replies = common_count = 0
    repetitive_turns = 0
    seen = previous.copy()

    def components(prefix_len):
        remaining = count - prefix_len
        suffix_common = 0
        if remaining:
            suffix_common = api.calculate_have_common_prefix(0, seen, api.NO_REPLY_STR)[
                1
            ]
            suffix_common += (remaining - 1) * api.calculate_have_common_prefix(
                0, [api.NO_REPLY_STR], api.NO_REPLY_STR
            )[1]
        return {
            "reward_pauc": sum(auc.values()) / len(auc) if auc else 0.0,
            "reward_repetition": (
                -repetition_sum / repetition_count if repetition_count else 0.0
            ),
            "reward_outspan": -out_replies / outside_count if outside_count else 0.0,
            "reward_common_prefix": (
                -(common_count + suffix_common) / count if count else 0.0
            ),
        }

    def total(parts):
        return (
            api.PAUC_WEIGHT * parts["reward_pauc"]
            + api.REPETITION_WEIGHT * parts["reward_repetition"]
            + api.OUTSPAN_WEIGHT * parts["reward_outspan"]
            + api.COMMON_PREFIX_WEIGHT * parts["reward_common_prefix"]
        )

    prefixes = [total(components(0))]
    for i, rec in enumerate(records):
        for scored in rec["similarity"]:
            wid = scored["window_id"]
            window = window_by_id[wid]
            q = scored["value"] / max_score
            time_weight = 1 - 0.5 * (rec["decision_time"] - window["start"]) / (
                window["end"] - window["start"]
            )
            auc[wid] += (q - quality[wid]) * time_weight
            quality[wid] = q
        for scored in rec["repetition"]:
            repetition_sum += int(scored["value"])
            repetition_count += 1
        repetitive_turns += int(any((scored["value"] for scored in rec["repetition"])))
        out_replies += int(outside[i] and rec["text"] != api.NO_REPLY_STR)
        common_count += api.calculate_have_common_prefix(
            rec["decision_time"], seen, rec["text"]
        )[1]
        seen.append(rec["text"])
        prefixes.append(total(components(i + 1)))
    return (prefixes, components(count))

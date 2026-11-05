HISTORY_SCOPE = "initial_rollout_messages_before_generation"


def assistant_history(messages):
    replies, indices, times = ([], [], [])
    for index, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else message.role
        if role != "assistant":
            continue
        content = (
            message.get("content") if isinstance(message, dict) else message.content
        )
        if isinstance(content, list):
            content = "".join(
                (part.get("text", "") for part in content if isinstance(part, dict))
            )
        if not isinstance(content, str):
            raise ValueError("Assistant scoring history must contain text")
        text = content.strip()
        if text and text != "NO REPLY":
            replies.append(text)
            indices.append(index)
            times.append(
                message.get("last_image_time")
                if isinstance(message, dict)
                else getattr(message, "last_image_time", None)
            )
    return {
        "previous_replies": replies,
        "history_scope": HISTORY_SCOPE,
        "history_message_indices": indices,
        "previous_reply_times": times,
    }


def scoring_ground_truth(reward_model, rollout=None):
    truth = dict(reward_model["ground_truth"])
    history = (
        rollout
        if rollout is not None and "previous_replies" in rollout
        else reward_model
    )
    if "previous_replies" in history:
        truth["previous_replies"] = list(history["previous_replies"])
        truth["prev_assistant_replies"] = list(history["previous_replies"])
        truth["history_scope"] = history.get("history_scope", HISTORY_SCOPE)
        truth["history_message_indices"] = list(
            history.get("history_message_indices", [])
        )
        candidates = list(reward_model.get("prev_assistant_replies") or [])
        truth["history_metadata_unmatched_count"] = sum(
            (
                str(reply).strip() not in truth["previous_replies"]
                for reply in candidates
            )
        )
        if (
            reward_model.get("previous_replies") == truth["previous_replies"]
            and reward_model.get("history_message_indices")
            == truth["history_message_indices"]
        ):
            truth["previous_reply_times"] = list(
                reward_model.get("previous_reply_times", [])
            )
    return truth

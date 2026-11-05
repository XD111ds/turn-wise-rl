import contextlib
import contextvars
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
import requests

PROTOCOL_VERSION = "credit-judge-final-v2-20261008"
_trace = contextvars.ContextVar("judge_trace", default=None)
_cache_context = contextvars.ContextVar("judge_cache_context", default=None)
logger = logging.getLogger(__name__)


def start_trace(context):
    return (_trace.set([]), _cache_context.set(context))


def finish_trace(tokens):
    events = _trace.get()
    _trace.reset(tokens[0])
    _cache_context.reset(tokens[1])
    return events


class InvalidAnswer(ValueError):
    pass


def final_answer(choice, answer_kind, max_scores):
    if not isinstance(choice, dict):
        raise InvalidAnswer("invalid_choice")
    if choice.get("finish_reason") not in ("stop", "eos"):
        raise InvalidAnswer("unfinished_or_missing_finish_reason")
    message = choice.get("message", {})
    if not isinstance(message, dict):
        raise InvalidAnswer("invalid_message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise InvalidAnswer("missing_final_content")
    content = re.sub("<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if answer_kind == "similarity":
        if content not in {str(i) for i in range(1, max_scores + 1)}:
            raise InvalidAnswer("invalid_similarity_format")
    elif answer_kind == "repetition":
        if content.lower() not in ("yes", "no"):
            raise InvalidAnswer("invalid_repetition_format")
        content = content.lower()
    else:
        raise ValueError(f"Unknown Judge answer kind: {answer_kind}")
    return content


@contextlib.contextmanager
def endpoint_slot(url, limit, timeout):
    root = Path(
        os.environ.get(
            "LLM_EVALUATOR_LOCK_DIR",
            str(Path(tempfile.gettempdir()) / f"turnwise-judge-locks-{os.getuid()}"),
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    endpoint = hashlib.sha256(url.encode()).hexdigest()
    deadline = time.monotonic() + timeout
    while True:
        for slot in range(limit):
            handle = (root / f"{endpoint}-{slot}.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("endpoint_concurrency_wait_timeout")
        time.sleep(0.05)


def _cache_path(key):
    root = os.environ.get(
        "LLM_EVALUATOR_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / f"turnwise-judge-cache-{os.getuid()}"),
    )
    if not root or root.lower() == "off":
        return None
    return Path(root) / key[:2] / f"{key}.json"


def _record(event):
    trace = _trace.get()
    if trace is not None:
        trace.append(event)


def validated_request(api, payload, headers, answer_kind):
    key_data = {
        "protocol": PROTOCOL_VERSION,
        "prompt_version": os.environ.get("LLM_EVALUATOR_PROMPT_VERSION", "original-v1"),
        "endpoints": sorted(api.URLS),
        "payload": payload,
        "kind": answer_kind,
        "max_scores": api.MAX_SCORES,
        "context": _cache_context.get(),
    }
    key = hashlib.sha256(
        json.dumps(key_data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    cache = _cache_path(key)
    if cache is not None:
        try:
            cached = json.loads(cache.read_text())
            if cached["key"] == key:
                answer = final_answer(
                    {"finish_reason": "stop", "message": {"content": cached["answer"]}},
                    answer_kind,
                    api.MAX_SCORES,
                )
                _record(
                    {
                        "kind": answer_kind,
                        "cache_key": key,
                        "cache_hit": True,
                        "attempt": 0,
                        "physical_request": False,
                        "elapsed_seconds": 0.0,
                    }
                )
                return answer
        except (OSError, ValueError, KeyError, TypeError):
            pass
    limit = int(os.environ.get("LLM_EVALUATOR_ENDPOINT_CONCURRENCY", "2"))
    if api.MAX_RETRIES < 1 or limit < 1 or api.REQUEST_TIMEOUT <= 0:
        raise ValueError(
            "Judge attempts, endpoint concurrency and timeout must be positive"
        )
    backoff = float(os.environ.get("LLM_EVALUATOR_RETRY_BACKOFF", "1"))
    for attempt in range(1, api.MAX_RETRIES + 1):
        url = api._next_url()
        started = time.monotonic()
        event = {
            "kind": answer_kind,
            "cache_key": key,
            "cache_hit": False,
            "endpoint": url,
            "attempt": attempt,
            "physical_request": False,
            "http_status": None,
            "finish_reason": None,
            "error_type": None,
        }
        answer = None
        try:
            with endpoint_slot(url, limit, api.REQUEST_TIMEOUT):
                event["queue_seconds"] = time.monotonic() - started
                event["physical_request"] = True
                response = api._get_session().post(
                    url, json=payload, headers=headers, timeout=api.REQUEST_TIMEOUT
                )
                event["http_status"] = response.status_code
                response.raise_for_status()
                choice = response.json()["choices"][0]
                event["finish_reason"] = (
                    choice.get("finish_reason") if isinstance(choice, dict) else None
                )
                answer = final_answer(choice, answer_kind, api.MAX_SCORES)
        except InvalidAnswer as exc:
            event["error_type"] = str(exc)
        except requests.RequestException as exc:
            event["error_type"] = type(exc).__name__
        except (ValueError, KeyError, IndexError, TypeError):
            event["error_type"] = "invalid_response_json"
        except TimeoutError:
            event["error_type"] = "endpoint_concurrency_wait_timeout"
        event["elapsed_seconds"] = time.monotonic() - started
        _record(event)
        if answer is not None:
            if cache is not None:
                temporary = None
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        mode="w", dir=cache.parent, delete=False
                    ) as handle:
                        temporary = handle.name
                        json.dump({"key": key, "answer": answer}, handle)
                    os.replace(temporary, cache)
                except OSError as exc:
                    logger.warning("Judge cache write failed: %s", type(exc).__name__)
                finally:
                    if temporary and os.path.exists(temporary):
                        os.unlink(temporary)
            return answer
        logger.warning(
            "Judge %s failed: endpoint=%s status=%s finish=%s error=%s attempt=%d/%d elapsed=%.2f",
            answer_kind,
            url,
            event["http_status"],
            event["finish_reason"],
            event["error_type"],
            attempt,
            api.MAX_RETRIES,
            event["elapsed_seconds"],
        )
        if attempt < api.MAX_RETRIES:
            time.sleep(max(0.0, backoff) * min(attempt, 3))
    return None

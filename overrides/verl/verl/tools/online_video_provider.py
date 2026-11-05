import logging
import json
import os
from typing import Any, Tuple
from verl.tools.base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from sglang.srt.function_call_parser import (
    ToolCallItem,
    FunctionCallParser,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OnlineVideoProvider(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        log_msg = f"Init {self.__class__.__name__} with config: {config}"
        logger.info(log_msg)
        self.annotation = dict()
        for path in config.get("annotation_paths", list()):
            self.annotation.update(json.load(open(path)))

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, dict]:
        question_id = kwargs.get("question_id", None)
        turn_id = parameters.get("turn_id", None)
        if turn_id is None:
            logger.error(f"parameters={parameters!r} should contain turn_id")
            return ("Invalid parameters", 0, None)
        if question_id is None:
            logger.error(f"kwargs={kwargs!r} should contain question_id")
            return ("Invalid parameters", 0, None)
        user_turns = self.annotation[question_id]
        if turn_id >= len(user_turns):
            return ("", 0, -1)
        next_user_turn = json.dumps(user_turns[turn_id]["content"])
        return (next_user_turn, 0, 0)


class ProvideOnlineVideoFunctionCallParser(FunctionCallParser):

    def __init__(self, tools):
        self.tools = tools

    def has_tool_call(self, content):
        return True

    def parse_non_stream(self, content, messages):
        num_user_turns_in_messages = len(
            [turn for turn in messages if turn.role == "user"]
        )
        tool_call_results = [
            ToolCallItem(
                tool_index=0,
                name="online_video_provider",
                parameters=json.dumps({"turn_id": num_user_turns_in_messages}),
            )
        ]
        return (content, tool_call_results)

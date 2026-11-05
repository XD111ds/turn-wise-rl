from __future__ import annotations
import asyncio
import logging
import multiprocessing as mp
import os
from contextlib import contextmanager
from copy import deepcopy
from json import JSONDecodeError
from typing import List, Optional, Tuple
from uuid import uuid4
import gc
import numpy as np
import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from sglang.srt.managers.tokenizer_manager import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.openai_api.protocol import Tool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    get_ip,
    get_open_port,
    is_cuda,
    maybe_set_triton_cache_manager,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolCall,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.net_utils import is_ipv6
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
)
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _set_envs_and_config(server_args: ServerArgs):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()
    set_ulimit()
    if server_args.tp_size * server_args.dp_size > 1:
        maybe_set_triton_cache_manager()
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at <FLASHINFER_INSTALLATION_URL>.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


class AsyncEngine(sglang.srt.entrypoints.engine.Engine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._need_reload = True

    async def release_memory_occupation(self):
        obj = ReleaseMemoryOccupationReqInput()
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def resume_memory_occupation(self):
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False
        obj = ResumeMemoryOccupationReqInput()
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

    async def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)
            ],
            load_format=load_format,
            flush_cache=flush_cache,
        )
        return await self.tokenizer_manager.update_weights_from_tensor(obj, None)

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()


def get_tool_call_parser_type(tokenizer: PreTrainedTokenizer) -> str:
    items = FunctionCallParser.ToolCallParserEnum.items()
    for parser_type, parser_cls in items:
        parser = parser_cls()
        if parser.bot_token.strip() in tokenizer.get_vocab() and (
            parser.eot_token == "" or parser.eot_token.strip() in tokenizer.get_vocab()
        ):
            return parser_type
    else:
        raise ValueError(f"No tool call parser found for tokenizer {tokenizer}")


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][
        0
    ]
    return prompt_token_ids[non_pad_index:].tolist()


class SGLangRollout(BaseRollout):

    def __init__(
        self,
        actor_module: str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        processor=None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.config.multi_turn.is_online_video_conversation = (
            self.config.multi_turn.get("is_online_video_conversation", False)
        )
        self._device_mesh_cpu = device_mesh
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        (
            self._tool_schemas,
            self._tool_map,
            self._tool_call_parser_type,
            self._sgl_tools,
            self._function_call_parser,
        ) = self._initialize_tools(config, tokenizer)
        assert not (
            not config.enforce_eager and config.free_cache_engine
        ), "disable CUDA graph (enforce_eager = False) if free cache engine"
        logger.info(
            f"tool_schemas: {self._tool_schemas}, tool_map: {self._tool_map}, tool_call_parser_type: {self._tool_call_parser_type}, sgl_tools: {self._sgl_tools}, function_call_parser: {self._function_call_parser}"
        )
        self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)
        self._verify_config(model_hf_config=model_hf_config)
        self._init_inference_engine(trust_remote_code, actor_module, port)
        self._init_sampling_params(**kwargs)
        self.tokenizer = tokenizer
        self.processor = processor
        self.pad_token_id = tokenizer.pad_token_id

    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert (
            self.tensor_parallel_size <= dist.get_world_size()
        ), "tensor parallel size should be less than or equal to the world size"
        self.train_tp = kwargs.get("train_tp", None)
        if self.train_tp is not None:
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // self.tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=self.tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )
        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )
            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)
        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        if self._rank == 0:
            logger.info(
                f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}"
            )
        visible_devices = [None] * self._device_mesh_cpu.size(1)
        torch.distributed.all_gather_object(
            visible_devices,
            os.environ["CUDA_VISIBLE_DEVICES"],
            self._device_mesh_cpu.get_group("tp"),
        )
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            sorted(list(self.visible_devices_set))
        )

    def _verify_config(self, model_hf_config):
        if not self.config.get("max_model_len", None):
            self.config.max_model_len = (
                self.config.prompt_length + self.config.response_length
            )
        assert (
            self.config.max_model_len
            >= self.config.prompt_length + self.config.response_length
        ), f"max_model_len should be greater than total sequence length (prompt_length + response_length):\n            {self.config.max_model_len} >= {self.config.prompt_length} + {self.config.response_length}"
        assert (
            model_hf_config.max_position_embeddings >= self.config.max_model_len
        ), "model context length should be greater than total sequence length"
        if self.config.multi_turn.max_turns is None:
            self.config.multi_turn.max_turns = self.config.max_model_len // 3
        if not self.config.get("max_new_tokens_per_generation", None):
            self.config.max_new_tokens_per_generation = self.config.response_length

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
            logger.info(f"running inference engine at: {dist_init_addr}")
        else:
            dist_init_addr = None
        load_format = (
            "dummy"
            if self.config.load_format.startswith("dummy")
            else self.config.load_format
        )
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0
        if first_rank_in_node:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = AsyncEngine(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                port=10000 + rank * 1100,
            )
        else:
            self._engine = None
        self.sharding_manager = None
        self.is_sleep = True

    def _init_sampling_params(self, **kwargs):
        kwargs = dict(
            n=1,
            max_new_tokens=self.config.response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
        )
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = self.config.get(k)
        self.sampling_params = kwargs

    def _initialize_tools(self, config, tokenizer):
        if config.multi_turn.tool_config_path is None:
            return ([], {}, None, [], None)
        import importlib.util
        import sys
        from omegaconf import OmegaConf
        from verl.tools.schemas import OpenAIFunctionToolSchema

        def initialize_tools_from_config(tools_config) -> list:
            tool_list = []
            for tool_config in tools_config.tools:
                cls_name = tool_config.class_name
                module_name, class_name = cls_name.rsplit(".", 1)
                if module_name not in sys.modules:
                    spec = importlib.util.find_spec(module_name)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = sys.modules[module_name]
                tool_cls = getattr(module, class_name)
                tool_schema_dict = OmegaConf.to_container(
                    tool_config.tool_schema, resolve=True
                )
                tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
                tool = tool_cls(
                    config=OmegaConf.to_container(tool_config.config, resolve=True),
                    tool_schema=tool_schema,
                )
                tool_list.append(tool)
            return tool_list

        tools_config_file = config.multi_turn.tool_config_path
        tools_config = OmegaConf.load(tools_config_file)
        tool_list = initialize_tools_from_config(tools_config)
        logger.info(f"Initialize tools from configuration.: tool_list: {tool_list}")
        tool_schemas = [
            tool.get_openai_tool_schema().model_dump() for tool in tool_list
        ]
        tool_map = {tool.name: tool for tool in tool_list}
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        if config.multi_turn.is_online_video_conversation:
            logger.info(
                "using a always return True `ProvideOnlineVideoFunctionCallParser` as dummy function call parser"
            )
            from verl.tools.online_video_provider import (
                ProvideOnlineVideoFunctionCallParser,
            )

            tool_call_parser_type = "always_true_tool_call_parser"
            function_call_parser = ProvideOnlineVideoFunctionCallParser(sgl_tools)
        else:
            tool_call_parser_type = get_tool_call_parser_type(tokenizer)
            function_call_parser = FunctionCallParser(sgl_tools, tool_call_parser_type)
        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {
            key: self.sampling_params[key]
            for key in kwargs
            if key in self.sampling_params
        }
        for key, value in kwargs.items():
            if key in self.sampling_params:
                self.sampling_params[key] = value
        try:
            yield
        finally:
            for key, value in old_sampling_params_args.items():
                self.sampling_params[key] = value

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        return self._req_level_generate_sequences(prompts, **kwargs)

    async def _async_rollout_a_request(
        self, req: AsyncRolloutRequest, do_sample: bool = True, **kwargs
    ) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None
        current_turns = 0
        if self.config.multi_turn.is_online_video_conversation:
            from verl.utils.reward_history import assistant_history

            question_id = _req.tools_kwargs["online_video_provider"]["execute_kwargs"][
                "question_id"
            ]
            first_turn = sum((msg.role == "user" for msg in _req.messages)) - 1
            total_turns = len(
                self._tool_map["online_video_provider"].annotation[question_id]
            )
            _req.credit_rollout = {
                "request_id": _req.request_id,
                "question_id": question_id,
                "expected_turn_ids": list(range(first_turn, total_turns)),
                "turns": [],
                "truncated": False,
                "end_reason": "unknown",
                **assistant_history(_req.messages),
            }
        while current_turns < self.config.multi_turn.max_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    tool_call_results = await asyncio.gather(
                        *[
                            self._tool_map[tool_call.function.name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                **_req.tools_kwargs[tool_call.function.name].get(
                                    "execute_kwargs", {}
                                ),
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )
                    if self.config.multi_turn.is_online_video_conversation:
                        messages_from_users = [
                            tool_call_result
                            for tool_call_result, tool_call in zip(
                                tool_call_results, parsed_tool_calls
                            )
                            if tool_call.function.name == "online_video_provider"
                        ]
                        tool_call_results = [
                            tool_call_result
                            for tool_call_result, tool_call in zip(
                                tool_call_results, parsed_tool_calls
                            )
                            if tool_call.function.name != "online_video_provider"
                        ]
                    _req.add_tool_response_messages(
                        self.tokenizer, [resp for resp, _, _ in tool_call_results]
                    )
                    for tool_call, (resp, reward, metrics) in zip(
                        parsed_tool_calls, tool_call_results
                    ):
                        _req.update_metrics(metrics, tool_call.function.name)
                    if self.config.multi_turn.is_online_video_conversation:
                        if messages_from_users[0][2] == -1:
                            _req.credit_rollout["end_reason"] = "input_exhausted"
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            break
                        user_turn_added = _req.add_user_response_messages(
                            self.processor, [resp for resp, _, _ in messages_from_users]
                        )
                        if not user_turn_added:
                            _req.credit_rollout.update(
                                truncated=True, end_reason="observation_budget"
                            )
                            finish_reason_type = FinishReasonTypeEnum.LENGTH
                            break
                    if len(_req.input_ids) >= _req.max_model_len:
                        if _req.credit_rollout:
                            _req.credit_rollout.update(
                                truncated=True, end_reason="context_budget"
                            )
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(
                        f"Unexpected tool calling last message state: {_req.messages[-1]}"
                    )
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                generation_prompt_ids = _req.get_generation_prompt_ids(self.tokenizer)
                if _req.get_generation_capacity(len(generation_prompt_ids)) < 1:
                    if _req.credit_rollout:
                        _req.credit_rollout.update(
                            truncated=True, end_reason="generation_budget"
                        )
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    break
                output = await self._handle_engine_call(_req, do_sample, **kwargs)
                content = output["text"]
                finish_reason_type = FinishReasonTypeEnum.from_str(
                    output["meta_info"]["finish_reason"]["type"]
                )
                if _req.credit_rollout:
                    _req.credit_rollout["current_generation_length_limited"] = (
                        finish_reason_type == FinishReasonTypeEnum.LENGTH
                    )
                current_turns += 1
                if (
                    finish_reason_type == FinishReasonTypeEnum.LENGTH
                    and _req.credit_rollout
                ):
                    _req.credit_rollout["truncated"] = True
                    _req.credit_rollout["generation_length_limited"] = True
                if finish_reason_type == FinishReasonTypeEnum.LENGTH and (
                    not self.config.multi_turn.is_online_video_conversation
                ):
                    _req.add_assistant_message(self.tokenizer, content)
                    break
                elif (
                    self._function_call_parser
                    and self._function_call_parser.has_tool_call(content)
                ):
                    finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                    _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                    try:
                        if not self.config.multi_turn.is_online_video_conversation:
                            normed_content, tool_calls = (
                                self._function_call_parser.parse_non_stream(content)
                            )
                        else:
                            normed_content, tool_calls = (
                                self._function_call_parser.parse_non_stream(
                                    content, _req.messages
                                )
                            )
                    except JSONDecodeError:
                        normed_content = content
                        tool_calls = []
                    except AttributeError:
                        normed_content = content
                        tool_calls = []
                    parsed_tool_calls = []
                    for tool_call in tool_calls:
                        function, has_decode_error = (
                            OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name, arguments=tool_call.parameters
                                )
                            )
                        )
                        if has_decode_error:
                            continue
                        parsed_tool_calls.append(
                            OpenAIFunctionToolCall(
                                id=str(tool_call.tool_index), function=function
                            )
                        )
                    if len(parsed_tool_calls) > 0:
                        _req.add_assistant_message(
                            self.tokenizer, normed_content, tool_calls=parsed_tool_calls
                        )
                    else:
                        _req.add_assistant_message(self.tokenizer, content)
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                        break
                else:
                    _req.add_assistant_message(self.tokenizer, content)
                    break
                gc.collect()
                get_torch_device().empty_cache()
        if current_turns >= self.config.multi_turn.max_turns:
            if _req.credit_rollout:
                _req.credit_rollout.update(truncated=True, end_reason="max_turns")
            finish_reason_type = FinishReasonTypeEnum.STOP

        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(
                _req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {})
            )
            await tool.release(
                _req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {})
            )
            return (name, reward)

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        _req.finalize(self.tokenizer, tool_reward_scores, finish_reason_type)
        if _req.credit_rollout:
            _req.credit_rollout["response_length"] = len(_req.response_ids)
        return _req

    async def _handle_engine_call(
        self,
        _req: AsyncRolloutRequest,
        do_sample: bool,
        override_n: bool = True,
        **kwargs,
    ) -> dict:
        generation_prompt_ids = _req.get_generation_prompt_ids(self.tokenizer)
        response_prompt_len = len(generation_prompt_ids) - len(_req.prompt_ids)
        max_new_tokens = min(
            self.config.max_new_tokens_per_generation,
            _req.get_generation_capacity(len(generation_prompt_ids)),
        )
        if max_new_tokens < 1:
            raise ValueError(
                f"Request {_req.request_id} has no generation capacity: input={len(generation_prompt_ids)}, response={response_prompt_len}, max_model={_req.max_model_len}, max_response={_req.max_response_len}"
            )
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=self.config.max_new_tokens_per_generation,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        kwargs["max_new_tokens"] = max_new_tokens
        if "n" not in kwargs or (kwargs["n"] > 1 and override_n):
            kwargs["n"] = 1
        if not self.config.multi_turn.is_online_video_conversation:
            with self.update_sampling_params(**kwargs):
                output = await self._engine.async_generate(
                    input_ids=generation_prompt_ids,
                    sampling_params=self.sampling_params,
                    return_logprob=False,
                )
        else:
            with self.update_sampling_params(**kwargs):
                output = await self._engine.async_generate(
                    prompt=_req.get_prompt(self.tokenizer),
                    image_data=_req.get_image_data(),
                    sampling_params=self.sampling_params,
                    return_logprob=False,
                )
        return output

    async def _handle_pending_state(
        self, _req: AsyncRolloutRequest
    ) -> AsyncRolloutRequest:
        if _req.tool_schemas is not None:
            tool_creation_coroutines = []
            for tool_schema in _req.tool_schemas:
                tool = self._tool_map[tool_schema.function.name]
                create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
                tool_creation_coroutines.append(
                    tool.create(_req.request_id, **create_kwargs)
                )
            await asyncio.gather(*tool_creation_coroutines)

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences_with_tools(self, prompts: DataProto, **kwargs) -> DataProto:
        logger.warning(
            "`generate_sequences_with_tools` is deprecated, please use `generate_sequences(...)`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._req_level_generate_sequences(prompts, **kwargs)

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", True)
        tgt_device = prompts.batch["input_ids"].device
        if self._tp_rank == 0:
            req_list = self._preprocess_prompt_to_async_rollout_requests(
                prompts, n=self.config.n
            )
            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(
                asyncio.gather(
                    *[
                        self._async_rollout_a_request(req, do_sample, **kwargs)
                        for req in req_list
                    ]
                )
            )
            sorted_output_req_list = sorted(
                output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset)
            )
        else:
            sorted_output_req_list = []
        dist.barrier()
        list_length = len(sorted_output_req_list) if self._tp_rank == 0 else 0
        [list_length] = broadcast_pyobj(
            data=[list_length],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        for i in range(list_length):
            sorted_output_req_list_batch = sorted_output_req_list[i : i + 1]
            [sorted_output_req_list_batch] = broadcast_pyobj(
                data=[sorted_output_req_list_batch],
                rank=self._rank,
                dist_group=self._device_mesh_cpu["tp"].get_group(),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            if self._tp_rank != 0:
                sorted_output_req_list.extend(sorted_output_req_list_batch)
        prompt_ids, response_ids = ([], [])
        prompt_attention_mask, response_attention_mask = ([], [])
        prompt_position_ids, response_position_ids = ([], [])
        prompt_loss_mask, response_loss_mask = ([], [])
        messages = []
        reward_scores = []
        multi_modal_inputs = []
        for req in sorted_output_req_list:
            assert (
                req.state == AsyncRolloutRequestStateEnum.COMPLETED
            ), f"Request {req.request_id} is not completed"
            assert (
                len(req.input_ids)
                == len(req.attention_mask)
                == len(req.position_ids)
                == len(req.loss_mask)
            ), f"Request {req.request_id} has different length of\n                len(req.input_ids)={len(req.input_ids)!r}, len(req.attention_mask)={len(req.attention_mask)!r}, len(req.position_ids)={len(req.position_ids)!r}, len(req.loss_mask)={len(req.loss_mask)!r}"
            error_message_lines = [
                f"Request {req.request_id} has input_ids length {len(req.input_ids)}\n                    greater than max_model_len {self.config.max_model_len}",
                f"Decoded input_ids: {self.tokenizer.decode(req.input_ids)}",
                f"Decoded prompt_ids: {self.tokenizer.decode(req.prompt_ids)}",
                f"Decoded response_ids: {self.tokenizer.decode(req.response_ids)}",
                f"Messages: {req.messages}",
                f"Max model length: {req.max_model_len}",
            ]
            error_message = "\n".join(error_message_lines)
            assert len(req.input_ids) <= self.config.max_model_len, error_message
            prompt_ids.append(
                torch.tensor(req.prompt_ids, dtype=torch.int, device=tgt_device)
            )
            response_ids.append(
                torch.tensor(req.response_ids, dtype=torch.int, device=tgt_device)
            )
            if len(req.response_ids) > self.config.response_length:
                logger.warning(
                    f"req.request_id={req.request_id!r} has response_ids length {len(req.response_ids)}\n                    greater than max_response_len {self.config.response_length},\nreq={req!r}"
                )
            prompt_attention_mask.append(
                torch.tensor(
                    req.prompt_attention_mask, dtype=torch.int, device=tgt_device
                )
            )
            response_attention_mask.append(
                torch.tensor(
                    req.response_attention_mask, dtype=torch.int, device=tgt_device
                )
            )
            prompt_position_ids.append(
                torch.tensor(
                    req.prompt_position_ids, dtype=torch.int, device=tgt_device
                )
            )
            response_position_ids.append(
                torch.tensor(
                    req.response_position_ids, dtype=torch.int, device=tgt_device
                )
            )
            prompt_loss_mask.append(
                torch.tensor(req.prompt_loss_mask, dtype=torch.int, device=tgt_device)
            )
            response_loss_mask.append(
                torch.tensor(req.response_loss_mask, dtype=torch.int, device=tgt_device)
            )
            messages.append({"messages": req.messages})
            reward_scores.append(req.reward_scores)
            multi_modal_inputs.append(
                {"pixel_values": req.pixel_values, "image_grid_thw": req.image_grid_thw}
            )
        prompt_ids = pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        if prompt_ids.shape[1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(
                prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True
            )
        response_ids = pad_sequence(
            response_ids, batch_first=True, padding_value=self.pad_token_id
        )
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(
                response_ids, self.config.response_length, self.pad_token_id
            )
        prompt_attention_mask = pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        if prompt_attention_mask.shape[1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(
                prompt_attention_mask, self.config.prompt_length, 0, left_pad=True
            )
        response_attention_mask = pad_sequence(
            response_attention_mask, batch_first=True, padding_value=0
        )
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(
                response_attention_mask, self.config.response_length, 0
            )
        prompt_position_ids = pad_sequence(
            prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
        )
        if prompt_position_ids.shape[1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(
                prompt_position_ids, self.config.prompt_length, 0, left_pad=True
            )
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=response_ids.device
        )
        delta_position_id = delta_position_id.unsqueeze(0).repeat(
            len(sorted_output_req_list), 1
        )
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
        prompt_loss_mask = pad_sequence(
            prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left"
        )
        if prompt_loss_mask.shape[1] < self.config.prompt_length:
            prompt_loss_mask = pad_sequence_to_length(
                prompt_loss_mask, self.config.prompt_length, 0, left_pad=True
            )
        response_loss_mask = pad_sequence(
            response_loss_mask, batch_first=True, padding_value=0
        )
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(
                response_loss_mask, self.config.response_length, 0
            )
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat(
            (prompt_attention_mask, response_attention_mask), dim=-1
        )
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            },
            batch_size=len(sorted_output_req_list),
        )
        if (
            self.config.free_cache_engine
            and self._engine is not None
            and (self._tp_rank == 0)
        ):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())
        return DataProto(
            batch=batch,
            non_tensor_batch={
                "messages": np.array(messages),
                "reward_scores": np.array(reward_scores),
                "credit_rollout": np.array(
                    [req.credit_rollout for req in sorted_output_req_list], dtype=object
                ),
                "multi_modal_inputs": np.array(multi_modal_inputs),
            },
        )

    def _preprocess_prompt_to_async_rollout_requests(
        self, prompts: DataProto, n: int
    ) -> list[AsyncRolloutRequest]:
        assert (
            "raw_prompt" in prompts.non_tensor_batch
        ), "need data.return_raw_chat=True, due to no official way do parse_messages"
        req_list = []
        for data_idx, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for rollout_offset in range(n):
                if self._tool_schemas:
                    _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                    _tool_schemas = [
                        self._tool_map[k].get_openai_tool_schema()
                        for k in _tools_kwargs.keys()
                    ]
                    _input_ids = None
                    _attention_mask = None
                else:
                    _input_ids = _pre_process_inputs(
                        self.pad_token_id, prompts.batch["input_ids"][data_idx]
                    )
                    _attention_mask = _pre_process_inputs(
                        0, prompts.batch["attention_mask"][data_idx]
                    )
                    _tools_kwargs = {}
                    _tool_schemas = None
                req = AsyncRolloutRequest(
                    batch_data_id=data_idx,
                    rollout_offset=rollout_offset,
                    request_id=str(uuid4()),
                    state=AsyncRolloutRequestStateEnum.PENDING,
                    messages=(
                        raw_prompt.tolist()
                        if not isinstance(raw_prompt, list)
                        else raw_prompt
                    ),
                    tool_schemas=_tool_schemas,
                    tools_kwargs=_tools_kwargs,
                    input_ids=_input_ids,
                    response_ids=[],
                    attention_mask=_attention_mask,
                    response_attention_mask=[],
                    response_position_ids=[],
                    response_loss_mask=[],
                    reward_scores={},
                    max_prompt_len=self.config.prompt_length,
                    max_response_len=self.config.response_length,
                    max_model_len=min(
                        self.config.max_model_len,
                        self.config.prompt_length + self.config.response_length,
                    ),
                    use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
                    enable_tokenization_sanity_check=self.config.multi_turn.enable_tokenization_sanity_check,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                )
                error_message = f"Request {req.request_id} has mismatched lengths: input_ids={len(req.input_ids)}, attention_mask={len(req.attention_mask)}, position_ids={len(req.position_ids)}, loss_mask={len(req.loss_mask)}"
                assert (
                    len(req.input_ids)
                    == len(req.attention_mask)
                    == len(req.position_ids)
                    == len(req.loss_mask)
                ), error_message
                req_list.append(req)
        return req_list

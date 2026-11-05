from collections import defaultdict
import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.utils.reward_history import scoring_ground_truth
import multiprocessing as mp
import os


@register("naive")
class NaiveRewardManager:

    def __init__(
        self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_processes = int(os.environ.get("LLM_EVALUATOR_NUM_PROCESSES", 20))

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompts_ids = []
        responses_ids = []
        prompt_lengths = []
        response_lengths = []
        data_sources = []
        ground_truths = []
        extra_infos = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompts_ids.append(valid_prompt_ids)
            responses_ids.append(valid_response_ids)
            prompt_lengths.append(valid_prompt_length)
            response_lengths.append(valid_response_length)
            data_sources.append(data_item.non_tensor_batch[self.reward_fn_key])
            ground_truths.append(
                scoring_ground_truth(
                    data_item.non_tensor_batch["reward_model"],
                    data_item.non_tensor_batch.get("credit_rollout"),
                )
            )
            extra_info = dict(data_item.non_tensor_batch.get("extra_info") or {})
            if "credit_rollout" in data_item.non_tensor_batch:
                extra_info["credit_rollout"] = data_item.non_tensor_batch[
                    "credit_rollout"
                ]
            extra_infos.append(extra_info)
        prompt_strs = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in prompts_ids
        ]
        response_strs = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in responses_ids
        ]
        print(f"geting reward with {self.num_processes} processes in parallel...")
        with mp.Pool(processes=self.num_processes) as pool:
            args = [
                (data_sources[i], response_strs[i], ground_truths[i], extra_infos[i])
                for i in range(len(data))
            ]
            scores = pool.starmap(self.compute_score, args)
        already_print_data_sources = {}
        for i in range(len(data)):
            score = scores[i]
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            if response_lengths[i] > 0:
                reward_tensor[i, response_lengths[i] - 1] = reward
            data_source = data_sources[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_strs[i])
                print("[response]", response_strs[i])
                print("[ground_truth]", ground_truths[i])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

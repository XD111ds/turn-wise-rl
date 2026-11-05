import hydra
import ray
from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.fs import copy_to_local


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    return RLHFDataset(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )


def create_rl_sampler(data_config, dataset):
    return SequentialSampler(dataset)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        OmegaConf.resolve(config)
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )
        role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec={"training": [config.trainer.n_gpus_per_node]},
            mapping={Role.ActorRollout: "training"},
        )
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=load_reward_manager(config, tokenizer, num_examine=0),
            train_dataset=create_rl_dataset(
                config.data.train_files, config.data, tokenizer, processor
            ),
            collate_fn=collate_fn,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    if (
        config.algorithm.adv_estimator != "grpo"
        or config.actor_rollout_ref.rollout.name != "sglang"
    ):
        raise ValueError("TurnWise requires GRPO with SGLang rollout")
    if config.trainer.nnodes != 1 or config.actor_rollout_ref.actor.strategy != "fsdp":
        raise ValueError("This entry point requires single-node FSDP training")
    if not ray.is_initialized():
        ray.init(
            num_cpus=config.ray_init.num_cpus,
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
            },
        )
    ray.get(TaskRunner.remote().run.remote(config))


if __name__ == "__main__":
    main()

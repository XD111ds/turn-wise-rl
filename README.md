# TurnWise

Core RL training code for **Turns Matter: Advancing Streaming Video RL via Turn-Level Credit**: silence-completed prefix rewards, turn-level credit assignment, and turn-mean policy loss. Alpha starts at **0**, increases linearly to **1.0** over the first **10%** of scheduled optimizer updates, and remains at **1.0** thereafter.

## Setup

Use Python 3.10 and a CUDA environment compatible with PyTorch 2.6.0. Prepare separate source checkouts of verl at `83ebd007e01de29bbe353de112d04245b4820b47` and SGLang at `dcae1fb2cd2def2a2eb4919c21f74e91d888d161`.

```bash
export VERL_PATH="<VERL_SOURCE_PATH>"
export SGLANG_PATH="<SGLANG_SOURCE_PATH>"
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python install.py --verl "$VERL_PATH" --sglang "$SGLANG_PATH"
```

The installer applies the training overrides to these checkouts and installs them in editable mode. The overrides are based on verl and SGLang, licensed under Apache-2.0; their original licenses and notices remain in the dependency checkouts.

## Training

Prepare the MMDuet2 RL training parquet, user-turn annotations, extracted frames, an SFT checkpoint compatible with Qwen2.5-VL, and an OpenAI-compatible chat-completions judge service. The paper uses gpt-oss-20b as the judge. All filesystem placeholders below must be replaced with absolute paths. `FRAME_ROOT` must contain the entries referenced under `./data/datasets/` in the annotations.

```bash
export MODEL_PATH="<SFT_CHECKPOINT_PATH>"
export TRAIN_PARQUET="<TRAIN_PARQUET_PATH>"
export RL_USER_TURNS_JSON="<USER_TURNS_JSON_PATH>"
export FRAME_ROOT="<FRAME_ROOT_PATH>"
export OUTPUT_DIR="<OUTPUT_PATH>"
export LLM_EVALUATOR_URL="<JUDGE_CHAT_COMPLETIONS_URL>"
export LLM_EVALUATOR_MODEL="<JUDGE_MODEL_NAME>"
bash train.sh
```

Defaults: one node with 8 GPUs, batch size 8, 4 rollouts per input, 90 frames at 2-second intervals, learning rate `1e-6`, one training epoch, frozen vision tower, and no KL or entropy regularization. Optional judge authentication uses `LLM_EVALUATOR_API_KEY`; multiple endpoints use comma-separated `LLM_EVALUATOR_URLS`. Hydra overrides can be passed after `bash train.sh`.

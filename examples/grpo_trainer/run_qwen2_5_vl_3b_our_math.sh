set -x
ENGINE=${1:-vllm}
project_name='MATH'
exp_name="grpo_0805"


# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training, 
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0
export WANDB_API_KEY=09395dec0812221d976e6fc6c165ba95070ed686


# MODEL_PATH=/gemini/platform/public/NLP/model/project_model/math/ckpts
CKPTS_DIR=/gemini/platform/public/NLP/model/project_model/math/${exp_name}

MODEL_PATH=/gemini/platform/public/NLP/private/zhaiyanbo/model/Qwen2.5-VL-3B-Instruct
# CKPTS_DIR=/gemini/platform/public/NLP/model/project_model/math/ckpts/${ckp_name}
TRAIN_FILE=/gemini/user/private/table_reasoning/zhaiyanbo/verl/data/rl_data/train.parquet  # ${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=/gemini/user/private/table_reasoning/zhaiyanbo/verl/data/rl_data/test.parquet  # ${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}"\
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_multi_modal_inputs=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='grpo_math_0803' \
    trainer.experiment_name='qwen2_5_vl_3b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto  $@ 2>&1 | tee log/${exp_name}.log
    
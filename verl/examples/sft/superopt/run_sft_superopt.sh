# Tested with 2 & 4 GPUs

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: verl/examples/sft/superopt/run_sft_superopt.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

train_path="superoptimize_c_v2_train_rl.parquet"
val_path="superoptimize_c_v2_val_rl.parquet"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_path \
    data.val_files=$val_path \
    data.prompt_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_key=extra_info \
    +data.response_dict_keys=['answer'] \
    +data.instruct_tuned=False \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=8 \
    data.max_length=4000 \
    data.truncation=right \
    model.partial_pretrain="Qwen/Qwen2.5-Coder-7B-Instruct" \
    trainer.default_local_dir=$save_path \
    trainer.project_name="superopt-sft" \
    trainer.experiment_name="superopt-sft" \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    trainer.total_epochs=5 
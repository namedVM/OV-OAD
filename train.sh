#! /usr/bin/bash
# 显卡 4, 5
export CUDA_VISIBLE_DEVICES=1,2,3,4

# 使用 nohup 启动，并将标准输出和错误重定向到 train.log
# 这样即使你关掉终端，任务也会继续，且你可以通过 tail 查看进度
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate ovoad
nohup torchrun \
--nproc_per_node=4 \
--master_port=29505 \
train.py --config configs/train.yml \
> train.log 2>&1 &

echo "训练已在后台启动，PID: $!"
echo "可以用指令查看日志: tail -f train.log"
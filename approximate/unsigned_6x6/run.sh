#!/usr/bin/env bash
set -euo pipefail

cd /home/xuanqi/tony/work/FPGA/approx_mlp/approx_66_mlp62

GPU_LIST=${GPU_LIST:-0}
NUM_RUNS=${NUM_RUNS:-8}
JOBS=${JOBS:-2}
OUT_ROOT=${OUT_ROOT:-runs_1to5_8runs}

./run_1to5_parallel.py \
  --num-runs "${NUM_RUNS}" \
  --jobs "${JOBS}" \
  --seed-base 0 \
  --seed-stride 1000 \
  --cuda-devices "${GPU_LIST}" \
  --out-root "${OUT_ROOT}"

# 含义：
# --num-runs 8：一共跑 8 组随机初始 INIT。
# --jobs 2：同时跑 2 组。
# --cuda-devices 0：指定跑在 GPU 0 上。
# --cuda-devices 0,1：并发 restart 会轮流分配到 GPU 0 和 GPU 1。
# --seed-base 0 --seed-stride 1000：第 0 组 seed=0，第 1 组 seed=1000，第 2 组 seed=2000。
# 第 1 步默认就是 --init-mode random。
# --random-init-prob 0.5 可以改随机 INIT 中 bit=1 的概率。
# --resume 可以断点续跑，已有最终输出的步骤会跳过。

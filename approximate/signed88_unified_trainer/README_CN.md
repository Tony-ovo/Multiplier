# 有符号 INT8 近似乘法器统一训练框架（第一版改进分支）

当前工程以 `train.py` 为唯一训练入口。V2 已移除；本文档描述现在实际使用的
NMSE、GEMM 累积 bias 和条件 bias 训练逻辑。

这套工程是按你上传的六类 RTL 与 W8A8 signed joint calibration CSV 重新整理的完整版本。

## 核心改变

### 1. 训练对象是完整 signed 8×8，而不是单独“训练 6×6 数据集”

训练数据直接来自 CSV 中的 signed INT8 `(a,b)`：

```text
signed a[7:0], signed b[7:0]
          ↓
完整近似 signed 8×8 模型
          ↓
approx_prod[15:0]
          ↓
与 exact = a*b 比较
          ↓
NMSE / GEMM累积bias / 条件bias / MAE / MRED / ER / zero loss
```

对你当前全部 RTL，近似只位于 `AL*BL`，signed 高位修正是精确的，因此严格满足：

```text
AL = a & 63
BL = b & 63
approx_prod = a*b + approx_LL(AL,BL) - AL*BL
```

代码为了速度，每个 epoch 只计算一次 4096 个 `(AL,BL)` 内部状态，然后按照 CSV 每一行的 signed `(a,b)` 索引对应的 LL 结果。**损失仍然逐 signed8 行计算**，尤其 MRED 的分母使用真实 `|a*b|`，因此 4096 状态只是缓存，不再是“训练数据定义”。

你上传的数据文件包含：

- 25,660 个 signed `(a,b)` 非零概率条目；
- `count` 总数 400,000；
- `p_calib` 总和 1；
- 4096 个 `(AL,BL)` 低位状态全部有覆盖；
- exact product 为 0 的 workload probability 约 0.154085。

### 2. 一个 Trainer 兼容所有硬件结构

训练器不再写：

```python
if design == "aggressive": ...
if design == "balanced": ...
```

硬件差异全部封装在 `signed88/hardware/designs/` 中。

```text
train.py
refine.py
verify.py
    │
    ▼
BaseDesign
    │
    ├── AggressiveDesign
    ├── CPHybridDesign
    │     ├── Default
    │     ├── Fast
    │     ├── Balanced
    │     └── Quality
    └── AreaDesign
```

当前真实 RTL 映射：

| design | 近似结构 | 可训练 INIT |
|---|---|---:|
| aggressive | 三个独立 6×2 + LUT-only compressor | 18 张表 |
| default | 三个 CP 6×2，使用 Default RTL wrapper | 4 张共享表 |
| fast | 三个 CP 6×2 | 4 张共享表 |
| balanced | 低/中 CP，最高 6×2 精确 | 4 张共享表 |
| quality | 仅最低 6×2 CP，其余精确 | 4 张表 |
| area | q16 三子块 + fixed low-carry-cut compressor | 2 张共享表 |

CP 类型保持原 RTL 设计约束：digit=0/1/2 的 truth-table 地址冻结，只训练 digit=3 对应的可达 INIT bit。

### 3. RTL 输出不是重新手写

训练结束以后，框架复制你上传的原始 RTL 目录，然后只替换 `DesignSpec` 明确声明的 LUT 实例 `.INIT(...)`。

因此：

- Fast 训练结果仍然生成 Fast 原 RTL；
- Default 会生成 Default 原 wrapper `signed88_approx`；
- Balanced/Quality 的精确 6×2、CARRY4 compressor、signed fused MAC 不会被修改；
- Area 的 compressor 不会被训练；
- Aggressive 的 18 张近似 truth table 可以独立训练。

## 目录

```text
signed88_unified_trainer/
├── train.py                 # 单次统一训练
├── refine.py                # 通用离散 bit/pair/basin refinement
├── verify.py                # Python 65536 signed 输入 + 可选 RTL 仿真
├── summarize.py             # 多 seed 汇总
├── analyze_calibration.py   # 查看 AI 数据分布
├── run_multi.sh             # 多 seed / 多 GPU
├── run_pipeline.sh          # train -> top-k -> refine
├── signed88/
│   ├── data.py              # signed joint histogram
│   ├── lut.py               # differentiable LUT6/LUT6_2 + STE
│   ├── losses.py            # signed8 最终输出 loss
│   ├── metrics.py           # hard workload/uniform metrics
│   └── hardware/
│       ├── base.py          # Design API、RTL patch、signed wrapper
│       ├── registry.py
│       └── designs/
│           ├── aggressive.py
│           ├── cphybrid.py
│           ├── area.py
│           └── common.py
├── rtl_sources/             # 你上传的原 RTL，作为模板保留
├── data/                    # 你上传的 calibration CSV
└── tests/
```

## 先运行回归测试

```bash
cd signed88_unified_trainer
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

当前测试已经验证：

1. 五种不同算术实现复现原 README baseline MAE/WCE/bias；
2. differentiable model 在 hard STE 前向下与 NumPy hard model 的全部 4096 LL 状态一致；
3. 所有 RTL `.INIT` binding 都可以被正确 patch。

## 单次训练

### Fast

```bash
python train.py \
  --design fast \
  --out-dir runs/fast_seed0 \
  --seed 0
```

### Default

```bash
python train.py --design default --out-dir runs/default_seed0 --seed 0
```

### Balanced

```bash
python train.py --design balanced --out-dir runs/balanced_seed0 --seed 0
```

### Aggressive

```bash
python train.py --design aggressive --out-dir runs/aggressive_seed0 --seed 0
```

### Area

```bash
python train.py --design area --out-dir runs/area_seed0 --seed 0
```

默认从 policy-approved mutable INIT bits 随机初始化：

```text
P(bit=1)=0.5
init_conf=0.55
```

若从原 RTL baseline 开始：

```bash
python train.py --design balanced --init-mode baseline --out-dir runs/balanced_baseline
```

若从之前结果继续：

```bash
python train.py \
  --design balanced \
  --init-mode json \
  --base-inits-json runs/balanced_seed0/best_signed88_inits.json \
  --out-dir runs/balanced_continue
```

## Curriculum

默认主训练：

### Stage 1

Soft LUT，主要使用低位 product-bit auxiliary supervision，同时 signed8 最终输出 MAE 提供数值约束。
GEMM NMSE/bias 主项先以 `0.02` 权重进入，避免初始 soft/hard gap 使大 bias
梯度完全盖住 bit warm-up。

### Stage 2

逐渐降低 bit loss：

```text
bit  1.00 -> 0.05
MAE  0.20 -> 0.02
MRED 0.0001 -> 0.0001
GEMM 0.02 -> 1.00
```

MRED 仍基于最终 signed8 `a*b` 计算，但只作为很弱的辅助梯度，避免小乘积
相对误差再次压过 MSE 与 bias。

### Stage 3

Stage 3 不再只是沿用 Stage 2 的连续 logit 做一段低学习率 Hard-STE。现在每轮：

1. 从当前**已经通过 hard 指标验收的全局 best INIT**重新建模；
2. 用 `init_conf=0.51` 将0/1位放到阈值两侧，并创建全新的 Adam，避免继承
   Soft 阶段的动量；
3. 做一个短 Hard-STE block。它只负责提供探索方向，不能直接证明某个 INIT 更好；
4. 围绕全局 hard best 精确计算候选 INIT 的完整离散硬件指标；
5. 只有 `objective_score` 严格下降的 single/pair move 才会写入 best JSON 和 RTL；
6. 接受后从新 best 再启动一轮；完整候选邻域没有改善时提前停止。

默认 `stage3-block-epochs=25`、`stage3-lr=2e-4`。对于
Default/Fast/Balanced/Quality 的56个可搜索位，每轮会完整评价：

```text
56 个 single-bit + C(56,2)=1540 个 pair-bit
```

梯度只决定检查顺序；默认 pair 范围覆盖全部56位，因此不会因为真实好位的梯度排名
靠后而漏掉。Area/Aggressive 的搜索空间更大，默认仍只取梯度排名靠前的64个single
和56个位组成pair，以控制成本；需要完整穷举时应按具体设计显式调大参数。
因此它们日志中的 `no_hard_progress` 只表示“当前截断候选集没有改善”，不能解释成
完整1/2-bit邻域最优。`summary.json` 会写出 `full_single_neighborhood` 和
`full_pair_neighborhood`，用来区分这两种情况。

相关参数：

```text
--stage3-restart-conf 0.51
--stage3-lr 0.0002
--stage3-block-epochs 25
--stage3-single-top-k 64       # 0 表示全部single
--stage3-pair-top-k 56         # 0 表示禁用pair
--stage3-pair-max-pairs 1540   # 0 表示不限制pair数量
--stage3-no-progress-rounds 1
```

主 Stage 3 的学习率现在由 `--stage3-lr` 独立控制；旧参数
`--stage3-lr-scale` 只保留给 population hard phase，不再控制主 Stage 3。

Hard-STE 轨迹即使临时翻到一个很差的硬配置，也不会污染最终artifact；最终接受始终由
精确 hard GEMM/NMSE/bias objective 决定。

### Population

从当前 hard best INIT 做很小的随机 bit perturb，再做局部 soft/hard tune，用来跳出 basin。

## 面向 GEMM 的误差与 bias 目标

当前默认目标不再让 MRED/ER 主导训练。设单次乘法误差
`e=approx-exact`，校准信号能量 `D=E[exact^2]`，则：

```text
NMSE = E[e^2] / D
global_bias = E[e]
global_bias_penalty = (K_eff-1) * global_bias^2 / D
```

为了避免“全局均值接近 0，但某些激活值或权重值仍持续产生同向误差”，
训练器还分别按 signed `a`、signed `b` 的 256 个取值计算条件均值误差，
并惩罚超过全局 bias 的条件 bias。默认 `K_eff=1024`；它表示代表性的
GEMM 内积长度，应按实际网络使用 `--bias-effective-k` 修改。

默认 soft loss 中：

```text
1.00 * normalized_MSE
+ 1.00 * global_bias_accumulation
+ 0.001 * conditional_bias_accumulation
+ small MRED / ER guardrails
```

这里三个主项只使用真实校准 workload。`--calibration-mix` 的 2% uniform
部分仍用于 MAE/MRED/ER/zero 等安全辅助项，不会悄悄改变 soft 与 hard 的
GEMM 主目标。

### Best 的选择

反向传播使用可微 surrogate，但最终 best 永远重新二值化 INIT 后用 hard evaluator 选：

```text
objective =
  1.00 * workload_NMSE
+ 1.00 * (K_eff-1) * workload_bias^2 / D
+ 0.001 * (K_eff-1) * conditional_bias_excess / D
+ 0.0001 * workload_MRED
+ 0.00005 * workload_ER
+ 0.05 * workload_NED
+ 0.0001 * uniform_MRED
```

soft loss、hard checkpoint、`refine.py` 和 `verify.py` 使用相同的
`K_eff` 与 bias 定义，避免训练得到低 bias 的 INIT 后又被旧 MRED score 淘汰。
新产物会写入 `objective_schema=gemm_nmse_bias_v1`；`refine.py`/`verify.py`
遇到旧产物时不会把旧版同名 `workload_bias` 权重误解释为新的平方 bias，
而是采用当前 GEMM 默认值并打印迁移提示。

## 多 seed

```bash
DESIGN=fast \
NUM_RUNS=8 \
JOBS=2 \
GPU_LIST=0,1 \
./run_multi.sh
```

### Default 全随机 logit 多 seed 实验

`run_default_random_logits.sh` 专门用于 Default 拓扑的全局探索。每个 seed
都会把**所有可训练 LUT INIT 位对应的 logit**分别从
`Normal(RANDOM_LOGIT_MEAN, RANDOM_LOGIT_STD)` 重新采样；不可训练的 LUT 位仍保持
RTL 基线常量。Default 当前是4张训练表中的56个 mutable logit（8+16+16+16）。
因此，这和先随机出 0/1 INIT、再用同一置信度生成 logit 并不相同。

建议第一轮固定同一种 logit 分布，只改变 seed，避免把初始化尺度与随机性的影响
混在一起：

```bash
cd /home/xuanqi/tony/work/FPGA/signed8x8_approx/signed88_unified_trainer

OUT_ROOT="$PWD/runs_random_logits/default_k4096_normal01_seed0_15" \
NUM_RUNS=16 \
JOBS=2 \
GPU_LIST=0,1 \
SEED_BASE=0 \
SEED_STRIDE=1 \
K_EFF=4096 \
RANDOM_LOGIT_MEAN=0.0 \
RANDOM_LOGIT_STD=1.0 \
./run_default_random_logits.sh
```

这里 `JOBS=2` 表示同时运行两个训练进程，两个 worker 分别绑定 GPU 0 和 1。
单张 GPU 建议从 `JOBS=1 GPU_LIST=0` 开始，以免显存不足。脚本默认运行16个
seed、`K_EFF=4096`、`RANDOM_LOGIT_MEAN=0`、`RANDOM_LOGIT_STD=1`。

全随机初始点的第一轮比较默认设置 `POPULATION_SIZE=0`。原因是每个 seed 已经是
独立的全局起点，而原 population 阶段会再执行24组、每组700 epoch 的单 bit
局部扰动，成本很高且不能替代 pair-bit 搜索。多 seed 汇总后，应选
`summary.csv` 排名前几名，再分别运行 `refine.py` 的 pair refinement。
如果确实希望同时运行 population，可显式设置 `POPULATION_SIZE=24`。

新版 Stage 3 已经内置 Default 的完整 single/pair hard 搜索。重新执行新的多seed
实验时必须使用新的 `OUT_ROOT`，不能覆盖旧结果。若只想继续优化已经跑完的公共终点，
无需重跑 Stage 1/2，可以直接执行。下面的正式结果目录已经生成；若要重复实验，
请把 `--out-dir` 换成新的目录名：

```bash
python3 train.py \
  --design default \
  --init-mode json \
  --base-inits-json runs_random_logits/default_k4096_normal01_seed0_15/overall_best_signed88_inits.json \
  --out-dir runs_random_logits/default_k4096_normal01_seed0_15/stage3_restart_exact_from_overall \
  --bias-effective-k 4096 \
  --stage1-epochs 0 \
  --stage2-epochs 0 \
  --stage3-epochs 100 \
  --population-size 0
```

这个命令的 `stage3-epochs=100` 是上限；完整邻域收敛后会提前停止。

例如先 refinement 汇总排名第一的产物：

```bash
python3 refine.py \
  --base-inits-json runs_random_logits/default_k4096_normal01_seed0_15/overall_best_signed88_inits.json \
  --out-dir runs_random_logits/default_k4096_normal01_seed0_15/refine_overall_best \
  --bit-rounds 20 \
  --pair-rounds 8 \
  --pair-candidate-bits 56 \
  --pair-max-pairs 1540 \
  --basin-iters 0
```

若要严格比较多个 basin，不应只 refinement 第一名；还应从 `summary.csv` 取前3～5个
不同 INIT，分别传给 `--base-inits-json`。`initSha256` 相同表示不同 seed 实际收敛到了
同一个硬 INIT，不需要重复做 PPL。最终再用同一组硬指标和外部 PPL 测试排序。

脚本还可以安全地把附加参数原样传给 `train.py`，例如缩短一次 smoke test：

```bash
NUM_RUNS=2 JOBS=1 GPU_LIST=0 K_EFF=4096 \
OUT_ROOT="$PWD/runs_random_logits/default_smoke" \
./run_default_random_logits.sh \
  --stage1-epochs 20 \
  --stage2-epochs 20 \
  --stage3-epochs 10
```

如果机器或进程中断，使用**完全相同的实验参数和输出目录**继续批任务：

```bash
OUT_ROOT="$PWD/runs_random_logits/default_k4096_normal01_seed0_15" \
NUM_RUNS=16 JOBS=2 GPU_LIST=0,1 \
SEED_BASE=0 SEED_STRIDE=1 K_EFF=4096 \
RANDOM_LOGIT_MEAN=0.0 RANDOM_LOGIT_STD=1.0 \
EXISTING_POLICY=resume \
./run_default_random_logits.sh
```

`resume` 会跳过已有完整 `summary.json`、JSON 和 RTL 的 seed；由于训练器没有保存
optimizer checkpoint，不完整的单次 run 会先被改名归档，再从该 seed 起点重新训练。
对带有本脚本 manifest 的实验目录，`EXISTING_POLICY=replace` 也采用改名归档而非
删除；manifest 同时记录 `train.py` 的 SHA256，因此修改训练算法后不能把旧run和新run
误混在同一次 resume 中。脚本拒绝 replace 一个没有自身 manifest 的非空目录。只要任意 seed 失败，脚本就返回
非零状态且不会生成本轮总汇；全部成功后才生成：

```text
experiment_config.json
planned_runs.json
run_000_seed_0/ ... run_015_seed_15/
summary.csv
summary.json
overall_best_signed88_inits.json
```

## 离散 refinement

```bash
python refine.py \
  --base-inits-json runs/fast_seed0/best_signed88_inits.json \
  --out-dir refine/fast_seed0
```

refiner 不知道 Fast/Area/Aggressive 内部结构，只调用：

```text
design.spec.search_bits
design.hard_low_numpy()
```

然后完成 single-bit、pair-bit、basin-hop。

默认会自动继承训练 JSON 中的 calibration 与 objective 权重。

## 验证

```bash
python verify.py \
  --inits-json refine/fast_seed0/best_signed88_inits.json \
  --rtl-dir refine/fast_seed0/best_rtl
```

Python verifier 会显式遍历：

```text
256 × 256 = 65,536
```

组 signed8 输入，并根据最终 signed16 输出验证误差。

如果安装 Icarus Verilog 与 Xilinx `cells_sim.v`：

```bash
python verify.py \
  --inits-json refine/fast_seed0/best_signed88_inits.json \
  --rtl-dir refine/fast_seed0/best_rtl \
  --run-rtl \
  --cells-sim /path/to/cells_sim.v
```

## 新增新的乘法器结构

以后再有一套完全不同的近似 RTL，不需要复制 `train_xxx.py`。

只新增一个 Design plugin：

```python
class NewDesign(BaseDesign):
    spec = DesignSpec(...)

    def build_core(self, inits, init_conf, noise_std):
        # 对应硬件结构的 differentiable PyTorch 模型
        ...

    def hard_low_numpy(self, inits):
        # 完全离散、与 RTL 一致的 4096-state hard model
        ...
```

并在：

```text
signed88/hardware/registry.py
```

注册一次。

`train.py / refine.py / verify.py / summarize.py / run_multi.sh` 全部不需要修改。

# 有符号 INT8 近似乘法器统一训练框架

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
MRED / ER / MAE / bias / zero loss
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

### Stage 2

逐渐降低 bit loss：

```text
bit  1.00 -> 0.05
MAE  0.20 -> 0.02
MRED 0.10 -> 1.00
```

这里 MRED 已经是基于最终 signed8 `a*b` 的 workload MRED。

### Stage 3

LUT 前向使用 STE 二值化，主要优化最终 signed8 输出误差，降低 soft/hard deployment gap。

### Population

从当前 hard best INIT 做很小的随机 bit perturb，再做局部 soft/hard tune，用来跳出 basin。

## Best 的选择

反向传播使用可微 surrogate，但最终 best 永远重新二值化 INIT 后用 hard evaluator 选：

```text
objective =
  1.00 * workload_MRED
+ 0.25 * workload_ER
+ 0.10 * workload_NED
+ 0.05 * |workload_bias| / 16384
+ 0.05 * uniform_MRED
```

## 多 seed

```bash
DESIGN=fast \
NUM_RUNS=8 \
JOBS=2 \
GPU_LIST=0,1 \
./run_multi.sh
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

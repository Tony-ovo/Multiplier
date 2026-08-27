# 面向 AI/GEMM 的有符号 INT8 近似乘法器训练框架

本文档是当前工程的中文主说明，依据现有 `train.py`、`signed88/`、`refine.py`、`verify.py` 和 RTL 模板整理。已经删除的 V2 不属于本文讨论范围。

这套框架的任务不是训练神经网络权重，而是训练 FPGA 近似乘法器中 LUT6/LUT6_2 的 `.INIT` 真值表。训练结束后，每个可训练参数必须重新变成 0/1，组成真正可综合的 64 位 INIT，并写回原 RTL。CARRY4、精确高位路径、冻结的 LUT 地址和整体硬件拓扑不会被训练修改。

一句话概括整个方法：

> 先把二值 LUT 临时松弛成可微分模型，用真实 AI 校准分布和 GEMM 误差代理获得梯度；再把 INIT 二值化，用完全离散的硬件模型严格选优；最后用 Stage3 的精确 single/pair bit 搜索弥补 soft-to-hard 差距并导出 RTL。

---

## 1. 全局流程

```text
signed INT8 联合直方图 CSV
        │
        ├── a、b、exact=a*b、样本概率
        └── 映射到 4096 个低6位状态 (AL, BL)
                         │
                         ▼
              可微的近似低6×6硬件模型
       LUT logit → sigmoid/sharp → soft LUT/STE
                         │
                         ▼
      approx = a*b + approx_LL(AL,BL) - AL*BL
                         │
                         ▼
        GEMM NMSE + 全局bias + 条件bias + 辅助loss
                         │
               loss.backward() + Adam
                         │
                         ▼
             LUT logit跨过0阈值，INIT位翻转
                         │
        定期阈值化，并用NumPy硬模型重新评价
                         │
                         ▼
 Stage1 soft warm-up → Stage2 soft GEMM ramp
                         │
                         ▼
 Stage3 Hard-STE重启 + 梯度排序 + 精确single/pair验收
                         │
                         ▼
            可选Population局部随机重启
                         │
                         ▼
       best INIT JSON → patch原RTL → 另行运行verify.py做65,536输入验证
```

这里存在两个必须分清的目标：

- `compute_loss()` 生成可微训练代理，用于反向传播。
- `evaluate_design()` 生成完全离散的 `objective_score`，用于保存 best、Stage3 精确验收、refine 和多 seed 排序。

两者的 GEMM NMSE、全局 bias 和条件 bias 定义相互对齐，但完整公式并不完全相同。因此，soft loss 下降不代表最终硬 INIT 一定更好，最终以 hard score 为准。

还要区分三组容易混淆的运行配置：

| 配置来源 | design/init | K | Stage1/2/3 | Population |
|---|---|---:|---|---:|
| 直接运行 `train.py` 的代码默认 | fast/random | 1024 | 6000/10000/500 | 24 |
| `run_default_random_logits.sh` 的脚本默认 | default/random_logits N(0,1) | 4096 | 未覆盖时仍为6000/10000/500 | 0 |
| 当前干净32-seed实验 | default/random_logits N(0,1) | 4096 | 6000/10000/100 | 0 |

本文讲公式时会注明代码默认；第15节的结果分析采用第三行的实际实验配置。

---

## 2. 代码结构与调用关系

### 2.1 目录职责

```text
signed88_unified_trainer/
├── train.py
│   └── 单次训练入口；初始化、Stage1/2/3、Population、保存best
├── refine.py
│   └── 不使用梯度的离散single/pair/basin后处理
├── verify.py
│   └── Python全输入验证、artifact核对、可选Icarus RTL仿真
├── summarize.py
│   └── 汇总多seed结果，按hard objective_score排序
├── run_default_random_logits.sh
│   └── Default随机logit多seed专用脚本，带manifest和批级resume
├── run_multi.sh
│   └── 旧的通用多seed脚本
├── run_pipeline.sh
│   └── 旧的train → top-k → refine组合脚本
├── analyze_calibration.py
│   └── 校准数据统计
├── signed88/
│   ├── common.py
│   │   └── 公共常量、Metrics、ObjectiveWeights、JSON/INIT工具
│   ├── data.py
│   │   └── CSV校验、概率归一、低状态折叠、Torch batch
│   ├── lut.py
│   │   └── TrainableLUT6/LUT6_2、sharp、soft lookup、STE
│   ├── losses.py
│   │   └── 可微训练loss
│   ├── metrics.py
│   │   └── 完全离散的workload/uniform指标和hard score
│   ├── hard_search.py
│   │   └── LUT位梯度归一、候选排序、精确single/pair搜索
│   └── hardware/
│       ├── registry.py
│       │   └── 六类design注册表
│       ├── base.py
│       │   └── DesignSpec、SignedLow6Model、RTL patch公共接口
│       └── designs/
│           ├── cphybrid.py
│           │   └── default/fast/balanced/quality
│           ├── area.py
│           │   └── area
│           ├── aggressive.py
│           │   └── aggressive
│           └── common.py
│               └── 可微固定加法器、压缩器和布尔节点
├── rtl_sources/
│   └── 各类乘法器的原始RTL模板
├── data/
│   └── AI校准直方图
└── tests/
    └── 模型等价、RTL导出、bias目标、Stage3、随机logit等回归测试
```

### 2.2 一次训练的源码调用链

```text
train.py::main
  ├── parse_args
  ├── hardware.registry::get_design
  ├── data::load_calibration_csv → data::to_torch
  ├── initial_inits → design.build_model
  ├── metrics::evaluate_design                  # 初始hard指标
  ├── train_phase                               # Stage1/Stage2
  │   ├── losses::compute_loss
  │   │   └── SignedLow6Model::forward_signed_rows
  │   │       └── 具体Core::forward_bits
  │   │           └── TrainableLUT6/LUT6_2::forward
  │   ├── loss.backward
  │   ├── clip_grad_norm_
  │   ├── Adam.step
  │   └── hard_inits → evaluate_design → save_best
  ├── Stage3
  │   ├── 从global hard best近阈值重建Hard-STE模型
  │   ├── 短Hard-STE block
  │   ├── rank_model_gradients
  │   └── exact_hard_step → hard NumPy single/pair验收
  ├── Population（可选）
  ├── BaseDesign::export_rtl
  └── 写入summary.json
```

### 2.3 为什么六套硬件能共用一个训练器

上层只依赖统一的 `BaseDesign` 接口：

- `build_model()`：建立可微 PyTorch 硬件模型。
- `hard_low_numpy()`：计算完全离散的 4096 个低位状态。
- `spec.mutable_bits/search_bits`：声明允许训练和搜索的 INIT 位。
- `spec.rtl_bindings`：声明每张训练表对应哪个 RTL 文件、模块和实例。
- `export_rtl()`：复制模板并只替换被声明的 `.INIT`。

| design | 可微/硬模型 | 低6×6结构 | INIT参数关系 | 可训练/搜索位 | 资源声明 |
|---|---|---|---|---:|---|
| `default` | `CPHybridDesign/Core` | 三个CP 6×2均近似，mask=`111` | 4张表在三段间共享 | 56 | 37 LUT6_2 + 6 CARRY4 |
| `fast` | `CPHybridDesign/Core` | 与Default算术相同，mask=`111` | 4张共享表 | 56 | 37 LUT6_2 + 6 CARRY4 |
| `balanced` | `CPHybridDesign/Core` | 低/中近似，高段精确，mask=`011` | 4张表由两个近似段共享 | 56 | 39 LUT6_2 + 7 CARRY4 |
| `balanced_split` | `CPSplitDesign/Core` | 与balanced算术相同，低/中近似 | lo/mid 各 4 张独立表（8 张） | 112 | 39 LUT6_2 + 7 CARRY4 |
| `quality` | `CPHybridDesign/Core` | 仅最低段近似，mask=`001` | 4张表仅低段使用 | 56 | 40 LUT6_2 + 8 CARRY4 |
| `area` | `AreaDesign/Core` | 三个q16子块 + 固定截断进位压缩 | 2张表由三段共享 | 128 | 29 LUT6_2 + 5 CARRY4 |
| `aggressive` | `AggressiveDesign/Core` | 三个独立6×2 + LUT-only压缩器 | 12张子块表 + 6张压缩表 | 896 | 31 LUT6_2 + 4 LUT6 + 4 CARRY4 |

“共享表”表示多个 RTL 实例受同一组 INIT 参数约束，不表示综合后只剩一个物理 LUT。一个共享 INIT 位改变时，所有使用该模块真值表的实例会一起变化。

`balanced_split` 是 `balanced` 的 INIT 解耦版本：原本 `b[1:0]`（权重 ×1）与 `b[3:2]`（权重 ×4）两个近似段被迫共用同一个 `s8862_approx62_cp` 模块的真值表；解耦后两个物理实例（本来就存在）各带独立 INIT（`s8862_approx62_cp_lo/mid`，模板在 `rtl_sources/BalancedSplit/`），资源完全不变，可训练位从 56 翻倍到 112，允许 ×1 段与 ×4 段学到不同的误差抵消策略。基线 INIT 与 balanced 相同，因此基线指标一致（uniform WCE=80）。

CP设计只开放 `digit=3` 对应的可达地址：

- `cp_lut01`：8位；
- `cp_lut23`、`cp_lut45`、`cp_lut67`：各16位；
- 总数：`8+16+16+16=56`。

`digit=0/1/2` 的地址、精确高位子块、固定压缩器和 signed fused MAC 不会被训练破坏。`normalize_inits()` 也会拒绝任何改动冻结位的 JSON。

---

## 3. 为什么 signed 8×8 可以缓存成 4096 个状态

对一个 signed INT8 输入对，定义：

```text
AL = a & 63
BL = b & 63
```

当前六类 RTL 只近似低位 `AL×BL`，上部有符号修正与交叉项保持精确。因此最终输出严格写成：

```text
approx_signed(a,b)
  = a*b + approx_LL(AL,BL) - AL*BL
```

所以乘法误差为：

```text
e(a,b) = approx_signed - a*b
       = approx_LL(AL,BL) - AL*BL
```

误差只取决于 64×64=4096 个 `(AL,BL)` 状态。代码每次 forward 先计算完整的 4096-state low grid，再根据每一行 CSV 的 `state_index` gather 对应结果。

这只是等价的计算缓存，并没有把训练任务偷换成 unsigned 6×6：

- loss 的 `a`、`b` 和 `exact=a*b` 仍是 signed INT8/INT16；
- workload MRED 的分母仍是真实 `|a*b|`；
- bias 和条件 bias 仍按真实 signed `a`、signed `b` 分组；
- `verify.py` 仍显式遍历全部 256×256=65,536 组 signed 输入。

在均匀 signed INT8 输入空间中，每个低6位状态对应16组完整 signed 输入，所以 uniform 指标也可以精确折叠到4096状态。

---

## 4. 校准数据如何进入训练

默认数据：

```text
data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv
```

CSV 至少需要：

- `a`：范围 `[-128,127]` 的 signed INT8；
- `b`：范围 `[-128,127]` 的 signed INT8；
- 一个正权重列。

`--calibration-weight-column auto` 的选择优先级为：

```text
count → p_calib → weight → probability
```

加载器还会检查：

- `(a,b)` 不重复；
- 权重有限且严格大于0；
- 数据非空；
- 操作数位于 signed INT8 范围。

当前默认文件的统计为：

- 25,660 个非零直方图桶；
- `count` 总和 400,000；
- 4096/4096 个低位状态都有覆盖；
- `P(exact=0)≈0.154085`。

每行原始权重会归一化成概率 `p_r`。每个 epoch 使用整个直方图 full batch，不进行 mini-batch 抽样或 shuffle。

`a` 与 `b` 的角色不能随意交换。当前数据是联合分布，硬件本身也可能非交换；在 AI 集成中必须始终保证训练时的 activation/weight 端口映射与实际 GEMM 调用一致。

### Workload 与 uniform 的关系

默认 `calibration_mix=0.98`：

- MAE、MRED、ER surrogate、zero loss 和 bit 辅助项使用 98% workload + 2% uniform 安全混合；
- 主 GEMM NMSE、全局 bias、条件 bias 只使用真实 workload；
- uniform NMSE/bias 虽会计算用于诊断，但不进入当前总 loss 的 GEMM 主项。

这样做是为了不让均匀分布悄悄改变面向真实 AI 数据的主目标，同时保留少量全输入域安全约束。

---

## 5. LUT INIT 如何变成可训练参数

### 5.1 Logit 参数化

每个可训练 INIT 位 `j` 对应一个实数 logit `z_j`：

```text
p_j = sigmoid(z_j)
```

训练期间 `p_j∈(0,1)`，导出时：

```text
hard_bit_j = 1[p_j >= 0.5] = 1[z_j >= 0]
```

冻结地址仍强制替换成原 RTL 的 0/1 常量，因此这些位置对 loss 的梯度为0，也不能在导出时改变。

### 5.2 初始化模式

| 模式 | 含义 |
|---|---|
| `random` | 先按 `random_p` 采样硬 0/1，再用共同 `init_conf` 映射到正负 logit |
| `random_logits` | 每个 mutable logit 独立采样 `Normal(mean,std²)`，冻结位不变 |
| `baseline` | 从原 RTL 基线 INIT 开始 |
| `json` | 从已有 artifact 的 INIT 开始 |
| `json_perturb` | 从已有 INIT 对 search bits 随机翻转，并保证至少改变一位 |

默认 `random` 使用：

```text
P(bit=1)=0.5
init_conf=0.55
logit(0.55)≈+0.20067
logit(0.45)≈-0.20067
```

`random_logits` 直接生成连续的不同置信度，适合多 seed 全局起点实验。该模式要求 `init_noise_std=0`，否则训练入口会直接拒绝配置。`initial_signed88_inits.json` 保存的是模型实际阈值化后的 hard INIT，而不是未经模型处理的 requested INIT。

### 5.3 sharp 函数与参数 c

代码会先把 `x` clamp 到 `[eps,1-eps]`；随后定义：

```text
S_c(x) = x^c / (x^c + (1-x)^c + eps)
```

忽略 `eps` 时等价于：

```text
S_c(x) = sigmoid(c * logit(x))
```

忽略分母中的浮点 `eps` 时：

- `c=1`：代码显式短路，返回 clamp 后的 `x`；
- `c>1`：前向值更靠近0/1；
- `0<c<1`：前向值更软；
- 理论分界仍为0.5；当 `c!=1` 时，实际分母额外加的 `eps` 会带来极小浮点偏移。

忽略 `eps`，其导数为：

```text
dS_c/dx = c*S_c(x)*(1-S_c(x)) / (x*(1-x))
dS_c(sigmoid(z))/dz = c*q*(1-q)
```

本次 `c=1` 时，最后一式就是常见的 `p*(1-p)`。

可训练表值为：

```text
q_j = S_c_init(sigmoid(z_j))
```

LUT 输出之后还会经过 `S_c_out`。当前代码默认：

```text
soft_c_init = soft_c_out = 1
hard_c_init = hard_c_out = 1
```

也就是说，当前实验没有依靠 sharp 自动锐化。`c>1` 只改变 soft forward 和局部梯度形状，并不会像正则项一样主动把原始 logit 推离0；过大的 `c` 还可能导致梯度饱和，尤其 `c_out` 会在 LUT、加法器和压缩节点中反复出现。真正显式的二值化正则是 `p(1-p)`，但当前 `bin_weight=0`。

如果以后研究 c 调度，应该把它作为独立消融变量，采用温和退火并同时监控 soft/hard gap、梯度范数和 INIT 翻转，而不能简单把 `c_init`、`c_out` 都固定成很大值。

### 5.4 可微 LUT 查找

对 n 输入 LUT，soft lookup 是真值表的多线性插值：

```text
LUT(x,q) = Σ_addr q_addr · Π_i [x_i, addr_i=1; 1-x_i, addr_i=0]
```

当输入全是0/1时，它精确选中一个 INIT 地址；当输入是连续值时，它相当于对所有地址进行可微加权。LUT6_2 中：

- O5 使用 `INIT[31:0]` 与 I0～I4；
- O6 使用完整64位 INIT 与 I0～I5。

---

## 6. Soft 训练与 Hard-STE 训练

### 6.1 Soft 模式

`hard_middle=False` 时：

- LUT 表值是连续值；
- LUT 输出是多线性插值；
- 中间加法、进位和压缩节点也是连续布尔多项式；
- 计算图平滑，适合大范围优化；
- 但 forward 不是最终部署的纯 0/1 电路，可能存在 soft/hard gap。

### 6.2 Hard-STE 模式

STE 定义为：

```python
hard = (x >= 0.5)
ste  = hard.detach() - x.detach() + x
```

它具有：

- 前向：`ste=hard`，所以中间信号是0/1；
- 反向：近似令 `d ste / d x = 1`，梯度继续传播。

`hard_middle=True` 时，每个 LUT 输入/输出和固定布尔节点的前向都会二值化。当前回归测试验证了 Hard-STE forward 与离散 NumPy 低位模型在4096状态上等价。但 STE 是有偏梯度代理，不能保证沿梯度更新后的离散 INIT 一定改善。

因此 Stage3 仍需要 exact hard evaluator。Hard-STE 可以产生候选并直接遇到更好的 hard checkpoint，也可以给离散 bit 排序；是否接受都必须重新看离散 `objective_score`。

---

## 7. 面向 GEMM 的主损失

### 7.1 记号

对 workload 中第 `r` 个输入对：

```text
y_r     = a_r * b_r
yhat_r  = 近似乘法结果
e_r     = yhat_r - y_r
p_r     = 归一化样本概率
D       = max(Σ p_r*y_r², 1)
mu      = Σ p_r*e_r
K       = bias_effective_k
```

当前默认 CSV 的信号能量约为：

```text
D ≈ 495600.5535625
```

### 7.2 NMSE

```text
workload_NMSE = Σ p_r*e_r² / D
```

MSE 对大误差进行平方惩罚，比 MRED 对小乘积的过度关注、以及 ER 对所有非零误差一视同仁，更接近 GEMM 数值扰动。

### 7.3 为什么必须惩罚全局 bias

对长度为 K 的点积，若各乘法误差近似独立同分布（i.i.d.）：

```text
E[(Σ e_k)²]
  = K*E[e²] + K*(K-1)*mu²
```

再除以 `K*D`，得到单项归一化代理：

```text
NMSE + (K-1)*mu²/D
```

因此全局 bias 项定义为：

```text
global_bias_penalty = (K-1)*mu²/D
```

即使单乘法 MSE 很低，只要 `mu` 不为0，GEMM 中同向误差仍会随 K 相干累积。这正是旧的 MRED/ER 主导目标容易遗漏的问题。

### 7.4 条件 bias

全局 `mu≈0` 仍可能是不同输入组之间正负抵消。例如某些固定权重值始终产生正误差，另一些始终产生负误差；在一个真实输出通道内部，这些误差未必会互相抵消。

对 signed `a` 的256个取值：

```text
P_a(g) = P(a=g)
mu_a(g) = E[e | a=g]
C_a = Σ_g P_a(g)*mu_a(g)²
```

对 signed `b` 同理得到 `C_b`。去除已经由全局 bias 解释的部分：

```text
C_excess = max(0, 0.5*(C_a+C_b)-mu²) / D
conditional_bias_penalty = (K-1)*C_excess
```

当前同时约束 A/B 两端，是因为单个联合直方图不能完整表示真实网络中每层、每通道的固定权重组合。这个条件项是结构化协方差的启发式代理，不等同于真实层级 dot trace。

### 7.5 GEMM 主项

可微训练中的 GEMM 部分是：

```text
L_gemm =
    mse_weight              * workload_NMSE
  + bias_weight             * global_bias_penalty
  + conditional_bias_weight * conditional_bias_penalty
```

当前默认：

```text
mse_weight              = 1.0
bias_weight             = 1.0
conditional_bias_weight = 0.001
```

随后每个 Stage 再用自己的 `gemm_weight` 乘整个 `L_gemm`。

`bias_effective_k` 的代码默认值是1024；最近面向长点积的 Default 实验显式使用4096。K 应根据实际 GEMM 内积长度设置，而不是越大越好。K 过大会让训练极度保守地追求零均值，K 过小则不能反映长点积漂移。

---

## 8. 其他辅助损失

以下记 `alpha=calibration_mix=0.98`，`mix(X)=alpha*X_workload+(1-alpha)*X_uniform`。

### 8.1 MAE

```text
W_MAE = E_workload[|e|] / 3969
U_MAE = E_uniform[|e|] / 3969
L_MAE = mix(MAE)
```

3969=`63²`，是精确 `AL*BL` 的最大值，用于按低6×6数值尺度归一化；它不是近似误差的严格上界，因此该 loss 不保证落在 `[0,1]`。

### 8.2 MRED

```text
W_MRED = E[|e|/|y| | y!=0]
L_MRED = mix(W_MRED,U_MRED)
```

MRED 现在只是极弱辅助项，避免再次让小乘积主导训练。

### 8.3 可微 ER surrogate

训练时不能直接对 `1[e!=0]` 求梯度，因此使用：

```text
ER_tau(e) = 1-exp(-|e|/tau)
L_ER = mix(E[ER_tau])
```

`tau` 按声明的 Stage1+Stage2+Stage3 总 epoch 从4线性调度到0.1。若 Stage3 提前停止，实际训练轨迹未必走到终点；Stage3 的 fresh ranking model 则固定使用 `tau=0.1`。它与 hard evaluator 中的真实 ER 并不相同；`tau=0.1` 对整数误差容易饱和，所以 ER 只保留很小权重。

### 8.4 零乘积辅助项

```text
W_zero = E[|e| | y=0] / 3969
L_zero = mix(W_zero,U_zero)
```

默认权重0.25，用于保护 AI 数据中较高的零值概率。它进入 soft loss，但当前不作为 hard score 的独立硬约束。

### 8.5 低积 bit 辅助监督

模型同时输出 `approx_LL` 的12个 bit。目标 bit 来自精确 `AL*BL`：

- soft阶段：先将预测 clamp 到 `[1e-6,1-1e-6]`，再计算逐 bit BCE；
- Hard-STE阶段：逐 bit L1，即 bit mismatch；
- state权重：98% workload low-state + 2% uniform；
- 默认位权：从 bit0 的1线性增至 bit11 的2，再归一化为均值1。

它的作用是帮助早期学习合理真值表，不是最终部署指标。

### 8.6 对称与二值正则

```text
L_sym = mean(|LL(AL,BL)-LL(BL,AL)|)/3969
L_bin = mean_over_tables(mean_over_mutable_bits(p*(1-p)))
```

当前默认：

```text
symmetry_weight = 0
bin_weight      = 0
```

也就是说，每张 LUT 先只对自己的 mutable bits 求均值，再对各 LUT 表等权平均；它并非把所有 mutable bits 全局等权平均。因此当前训练不会显式强迫交换对称，也不会靠 `L_bin` 把 logit 推离阈值。

### 8.7 完整 soft loss

```text
L_total =
    w_bit  * L_bit
  + w_mae  * L_MAE
  + w_mred * L_MRED
  + w_gemm * L_gemm
  + 0.0001 * L_ER
  + 0.25   * L_zero
  + 0      * L_sym
  + 0      * L_bin
```

这里的 `w_bit/w_mae/w_mred/w_gemm` 随 Stage 改变。

---

## 9. 梯度如何回传到 INIT

每个 epoch 的真实执行顺序：

```python
optimizer.zero_grad(set_to_none=True)
loss, terms = compute_loss(...)
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

默认 Stage1/2 使用 Adam，学习率 `0.002`。完整梯度链为：

```text
L_total
  → signed误差e
  → approx_LL数值及12个输出bit
  → 可微加法/压缩网络
  → LUT输出S_c_out（Hard时再经过STE，反向近似直通）
  → soft LUT地址插值
  → LUT连续表值q=S_c_init(sigmoid(logit))
  → 每个mutable INIT logit
```

几个主项的直观梯度：

```text
d NMSE / d e_r = 2*p_r*e_r/D
```

所以 MSE 把大误差拉得更强。

```text
d global_bias_penalty / d e_r = 2*(K-1)*mu*p_r/D
```

所以只要整体均值偏正，所有样本都会收到向负方向纠偏的共同梯度；均值偏负则相反。

条件 bias 则根据当前 `a_r` 和 `b_r` 所在分组的条件均值，给不同组施加不同方向的纠偏。当 `max(...)` 分支处于有效区间时：

```text
d conditional_bias_penalty / d e_r
  = (K-1)*p_r*(mu_a(a_r)+mu_b(b_r)-2*mu)/D
```

若条件 excess 被 clamp 为0，该分支梯度也为0。

冻结 INIT 位在 `table()` 中被常量替换，不参与最终输出，梯度为0。hard NumPy evaluator、`hard_inits()` 和 RTL patch 都不在 autograd 图中，只负责离散评价与部署。

训练器每隔默认25 epoch：

1. 将当前 logit 按0阈值变成 hard INIT；
2. 用 `hard_low_numpy()` 计算真实离散低位电路；
3. 用 `evaluate_design()` 计算 hard score；
4. 只有 score 严格下降时才更新 `best_signed88_inits.json` 和 `best_rtl/`。

因此 summary 中 `best_stage=initial` 并不表示没有发生梯度更新，而表示所有训练期间遇到的 hard INIT 都没有击败初始 hard score。

---

## 10. 每个 Stage 在做什么

### 10.1 Stage1：Soft bit warm-up

默认：

```text
epochs      = 6000
hard_middle = False
lr          = 0.002
bit         = 1.00
MAE         = 0.25
MRED        = 0
GEMM        = 0.02
```

完整形式：

```text
L_stage1 = L_bit + 0.25*L_MAE + 0.02*L_gemm
         + 0.0001*L_ER + 0.25*L_zero
```

主要目标：

- 先让 LUT 学会合理的低6×6输出位；
- 用 MAE 保持数值方向；
- 只少量引入 GEMM 项，避免随机 soft 电路一开始的巨大 bias 梯度压过 bit warm-up；
- 不关注 MRED。

Stage1 是平滑探索阶段，不应把其 soft loss 当作部署性能。

### 10.2 Stage2：Soft signed/GEMM ramp

默认10000 epoch，仍为 soft forward。令 `t=local_epoch/(epochs-1)`：

```text
bit  : 1.00   → 0.05
MAE  : 0.20   → 0.02
MRED : 0.0001 → 0.0001
GEMM : 0.02   → 1.00
```

即逐渐从“模仿准确低积 bit”转向“优化真实 signed workload 的 NMSE 和 bias”。Stage1、Stage2 共用同一个模型和 Adam，不在边界处重置参数或动量，且固定跑满，没有 early stop。

随机 logit 多 seed 曾从完全不同的初始 hard INIT 收敛到相同或少数几个硬吸引子。与这一现象一致的可能因素包括：

- 使用同一 full-batch 数据；
- 没有 dropout/mini-batch 噪声；
- 目标与拓扑相同；
- 长时间确定性 Adam 会把多个初值拉入同一 soft basin；
- soft 模型可以用分数值抵消 bias，但阈值化后这种补偿可能消失。

这些是机制解释，并非已经逐项完成因果消融后的证明。

多 seed 收敛一致说明吸引域稳定，不等于这个硬解是全局最优。

### 10.3 Stage3：Hard-STE + 精确离散搜索

旧 Stage3 只是在 Stage2 后继续小学习率 Hard-STE，logit 往往跨不过阈值。当前 Stage3 被设计成“连续提案 + 离散验收”的循环。

`train.py` 的 Stage3 总预算默认是500 epoch；当前32-seed实验显式设为100。Hard-STE训练代理的默认权重为：

```text
bit  = 0.005
MAE  = 0.04
MRED = 0.0001
GEMM = 1.0
lr   = 0.0002
block_epochs       = 25
no_progress_rounds = 1
```

每轮步骤如下：

1. 取当前已经验收的 global hard best。
2. 在首次进入、接受新 best 后，或丢弃一个未被接受且已发生 hard 漂移的模型后，用 `restart_conf=0.51` 重建：1位映射到0.51，0位映射到0.49，对应 logit 约 `±0.0400`。
3. 重建时新建 Adam，默认 `stage3_lr=2e-4`。
4. 运行最多25 epoch 的 Hard-STE block。
5. block 内定期阈值化；如果某个 Hard-STE 中间 INIT 的 hard score 更低，它可以直接成为 best。
6. 另外从当前 global best 建立 fresh ranking model，以 Hard-STE loss 反传一次。
7. 将 `dL/dlogit` 除以 table 对 logit 的局部导数，近似得到 `dL/d离散LUT位`。
8. 按预计翻位收益排序 single/pair 候选。
9. 每个候选都用 NumPy hard model 重新计算完整 score。
10. 只接受全体候选中严格优于当前 base 的最佳 move。
11. 接受后围绕新 best 重启并清空旧 Adam 动量；无进展达到 patience 后提前停止。

默认 `no_progress_rounds=1`，所以一次无改善就终止。若显式设为大于1，并且上一 block 没有跨过任何 hard 阈值，代码会保留同一模型与 Adam 继续下一 block，让 logit/动量累计；若模型漂移到一个未被接受的 hard 配置，则丢弃该轨迹并围绕 global best 重建。

对一个当前 hard 位 `h∈{0,1}`，翻位方向为：

```text
delta_bit = 1-2*h
predicted_delta ≈ gradient_bit * delta_bit
predicted_gain  = -predicted_delta
```

梯度只决定候选顺序或截断范围，最终接受不相信这个近似值。

#### Default/Fast/Balanced/Quality 的默认搜索范围

这些 CP 设计共有56个 search bits：

```text
56个single + C(56,2)=1540个pair + 1个base
= 每轮1597次hard评价
```

默认参数：

```text
--stage3-single-top-k 64       # 实际覆盖全部56；0也表示全部single
--stage3-pair-top-k 56         # 0表示禁用pair
--stage3-pair-max-pairs 1540   # 0表示pair不设上限
```

完整 pair 搜索非常重要，因为离散真值表会出现协同：两个 bit 单独翻都变差，同时翻却变好。

#### Area/Aggressive 的默认搜索范围

- Area：128个 search bits，完整 pair 数8128；
- Aggressive：896个 search bits，完整 pair 数400,960。

默认仍只评价前64个single和前56个位形成的最多1540个pair。因此其 `no_hard_progress` 只表示截断候选集无改善。

若确实要完整扫描，Area 可显式设置：

```bash
--stage3-single-top-k 0 \
--stage3-pair-top-k 128 \
--stage3-pair-max-pairs 0
```

Aggressive 则把 `128` 改为 `896`。这里仅在 Stage3 中，`single-top-k=0` 表示全 single，`pair-top-k=0` 表示禁用 pair，`pair-max-pairs=0` 表示 pair 数量不设上限。

Aggressive 的完整40万对每轮成本很高，应先做全 single，再按资源预算决定 pair 范围。

#### no_hard_progress 的准确含义

它不等于全局最优。只有满足以下全部条件，才能称“最终点是 Hamming 半径2局部最优”：

- 最后一轮围绕最终 INIT；
- single 与 pair 都完整覆盖；
- 最后一轮没有改善；
- 后续 Population 没有再次改写 best。

即使满足，也只排除了1位和2位邻居，不能排除3位以上协同或其他 basin。

### 10.4 Population：单次 run 内的局部随机重启

`train.py` 代码默认：

```text
population_size        = 24
population_epochs      = 700
population_soft_epochs = 150
population_flip_p      = 0.0007
population_lr          = 0.00025
population hard lr     = 0.00025*0.03 = 7.5e-6
```

每个 member：

1. 从当时的 global best 对 search bits 做小概率扰动，并强制至少翻一位；
2. 先做 soft tune；
3. 再做 Hard-STE tune；
4. 中间 hard checkpoint 若更好，可以更新 global best。

成员按顺序执行，所以后一个 member 可能围绕前一个成员刚发现的新 best，而不是24个固定锚点上的独立样本。Population 不做 exact pair 搜索，`population_summary.json` 保存的是每个成员的最终态；真正 global best 可能来自成员的中间 epoch。

`run_default_random_logits.sh` 默认把 `POPULATION_SIZE=0`，因为每个 seed 已经是独立全局起点，而当前 Stage3 已完整搜索 CP 的 single/pair 邻域。

### 10.5 Multi-seed、Population、Refine 的区别

| 方法 | 起点 | 是否用梯度 | 是否精确pair | 主要目的 |
|---|---|---:|---:|---|
| Multi-seed | 多个独立随机全局初始化 | 是 | 由各run的Stage3决定 | 测试初始化稳定性、寻找不同basin |
| Population | 单次run内当前best附近扰动 | 是 | 否 | 局部随机重启 |
| Stage3 exact | 当前global best邻域 | 梯度只排序 | 是 | 修复soft/hard gap和bit协同 |
| `refine.py` | 已有JSON | 否 | 是 | 完全离散的后处理与basin探索 |

多 seed 增加的是优化轨迹数量，不会增加 CSV 的400,000条原始计数，也不能代替独立验证集或 PPL 测试。

---

## 11. Hard score 如何定义 best

hard evaluator 先计算4096-state 完全离散输出，再得到 workload 与 uniform 指标。当前默认：

```text
objective_score =
    1.0     * workload_NMSE
  + 1.0     * global_bias_penalty
  + 0.001   * conditional_bias_penalty
  + 0.0001  * workload_MRED
  + 0.00005 * workload_ER
  + 0.05    * workload_NED
  + 0.0001  * uniform_MRED
  + w_wce   * uniform_WCE / 16384      # 默认 w_wce=0，--score-wce-weight 开启
```

其中：

```text
workload_NED = workload_MAE / 16384
```

`uniform_WCE` 项（`--score-wce-weight`，对应 `ObjectiveWeights.uniform_wce`）
是面向部署的新增项：PPL 对照实验显示全输入最坏误差档位（WCE tier）与
LLM PPL 退化的相关性远强于 NMSE，因此推荐部署导向的训练把它设为
1.0～4.0。默认 0 保证旧 artifact 分数逐位不变；`train.py`、`verify.py`、
`refine.py` 均支持该权重并会写入/继承 artifact 的 `objective_weights`。
soft 侧的对应物是 `--wce-weight`/`--wce-beta`（`logsumexp(beta*|e|)/beta`
的平滑最大值 surrogate，作用于全部 4096 个 uniform LL 状态），让梯度阶段
也感知极值误差，而不是只靠 Stage3 离散搜索去修。

注意 hard score 与 soft loss 的差异：

- hard ER 是严格的 `P(e!=0)`，soft ER 是指数 surrogate；
- hard NED 除以16384，soft MAE 除以3969；
- hard WCE 是精确 max，soft WCE 是 logsumexp 上界；
- bit、zero、symmetry、bin 不进入 hard score；
- hard score 没有默认 WCE 硬约束（只有加权项）；
- `refine.py` 可额外指定 WCE、NMSE、MRED、ER、绝对 bias 和条件 bias RMS 上限。

每个 checkpoint 只有满足：

```text
new_score < old_score - 1e-15
```

才会覆盖 best artifact。

常用指标含义：

| 字段 | 含义 |
|---|---|
| `workload_MSE/NMSE` | 校准分布单乘法平方误差及归一化值 |
| `workload_bias` | 校准分布有符号平均误差 `mu` |
| `workload_conditional_bias_a/b_rms` | 按 signed a/b 分组的条件均值误差RMS |
| `workload_gemm_NMSE` | 当前K下 NMSE + global/conditional bias代理 |
| `workload_predicted_dot_RMSE` | i.i.d.公式下长度K点积误差RMSE，仅含全局bias |
| `workload_bias_drift_sigma` | `sqrt(K)*mu/workload_RMSE`，相干均值漂移相对单乘法RMSE的带符号代理 |
| `workload_ER/MRED/MED` | workload真实硬ER、MRED、MAE |
| `ER/MRED/MED/WCE` | 全65,536 signed输入的uniform指标 |

`objective_schema=gemm_nmse_bias_v1` 用于区分当前平方 bias 语义和旧 artifact 的绝对 bias 语义。`refine.py/verify.py` 遇到旧 schema 会提示并使用当前默认目标，不会静默误解旧权重。

---

## 12. Refine 在做什么

`refine.py` 完全不使用 PyTorch 梯度，只使用离散 hard model：

1. 多轮扫描所有 single bit，贪心接受最佳严格改进；
2. 依据 single delta 排序，构造 pair pool并精确评价 pair；
3. 可选随机翻2～4位进入新 basin，再做 single polish；
4. 可设置硬约束：uniform WCE、workload MRED/ER/NMSE、绝对 bias、条件 bias RMS。

需要注意：

- 默认 pair 对 Area/Aggressive 仍可能被候选数限制；
- `refine.py --pair-max-pairs 0` 会得到零个 pair 候选，相当于禁用 pair；这与 Stage3 中同名参数的“不设上限”语义不同；
- 当前 `--neutral-bits` 不会实质改变 pair pool，因为完整 ranked 列表已经先占满候选；
- basin `steps.json` 的 `flips` 只记录初始扰动，不包含随后 polish 的完整净翻位；最终 JSON/metrics 仍是真实结果；
- refine 完成不自动等于全局最优，甚至不一定等于最终点完整半径2局部最优。

---

## 13. 产物与可追溯性

一个训练目录包含：

| 文件 | 内容 |
|---|---|
| `terminal_log.txt` | train.py 自身控制台输出 |
| `pipeline.log` | 只有通过多seed wrapper 启动时才有；保存wrapper捕获的完整进程输出 |
| `initial_signed88_inits.json` | 实际初始hard INIT和初始指标 |
| `best_signed88_inits.json` | 当前最优hard INIT、指标、训练参数和数据metadata |
| `best_rtl/` | 从模板复制并patch后的可部署RTL |
| `best_rtl/trained_artifact.json` | RTL内INIT与metadata |
| `history.jsonl` | 有评估事件时生成；记录定期hard checkpoint及Stage3离散事件，不是每个epoch一行 |
| `population_summary.json` | 各Population成员最终态 |
| `summary.json` | 单run总览、初始/best指标、Stage3统计、参数 |

训练器发现输出目录已有受保护 artifact 时会拒绝覆盖，必须使用新 `--out-dir`。

多 seed 汇总目录还包含：

```text
experiment_config.json
planned_runs.json
run_000_seed_xxx/ ...
summary.csv
summary.json
overall_best_signed88_inits.json
```

`summary.csv` 的 `initSha256` 是最终 best INIT mapping 的哈希，不是初始随机 INIT 的哈希。只有在同一 design、同一 RTL 模板的前提下，哈希相同才表示 INIT 配置相同；当前 Default 多seed实验可据此避免重复 PPL。该哈希不包含 design、RTL模板或 objective，不能跨设计推断电路相同。汇总器只复制 overall best JSON，不复制 overall RTL；应根据该行的 `best_rtl` 路径取得对应 RTL。

专用多 seed manifest 记录 `train.py` 的哈希，但没有记录所有 imported `signed88/*.py` 的哈希，也只记录 CSV 路径而非内容哈希。因此一批任务运行期间不要修改训练代码或 CSV；修改后必须使用新的 `OUT_ROOT`。

---

## 14. 推荐运行命令

先进入目录：

```bash
cd /home/xuanqi/tony/work/FPGA/signed8x8_approx/signed88_unified_trainer
```

### 14.1 回归测试

```bash
python3 -m unittest discover -s tests -v
```

### 14.2 从原 RTL baseline 训练 Default，K=4096

```bash
python3 train.py \
  --design default \
  --init-mode baseline \
  --bias-effective-k 4096 \
  --stage3-epochs 100 \
  --population-size 0 \
  --out-dir runs_gemm_bias/default_baseline_new
```

### 14.3 单个随机 logit 起点

```bash
python3 train.py \
  --design default \
  --init-mode random_logits \
  --random-logit-mean 0 \
  --random-logit-std 1 \
  --bias-effective-k 4096 \
  --seed 0 \
  --population-size 0 \
  --stage3-epochs 100 \
  --out-dir runs_random_logits/default_seed0_new
```

### 14.4 只对已有 JSON 运行新版 Stage3

```bash
python3 train.py \
  --design default \
  --init-mode json \
  --base-inits-json /path/to/best_signed88_inits.json \
  --bias-effective-k 4096 \
  --stage1-epochs 0 \
  --stage2-epochs 0 \
  --stage3-epochs 100 \
  --population-size 0 \
  --out-dir runs_random_logits/default_stage3_from_json_new
```

对于 Default 的56位，当前默认 Stage3 参数已经完整覆盖 single/pair；无需再额外写 `top-k`。

### 14.4b 面向部署的 balanced_split + WCE 训练（推荐）

`balanced_split` 有 112 个搜索位，默认 Stage3 预算（single 64 / pair 56 /
1540 对）覆盖不全，需要显式放大到全预算；同时打开 WCE 项：

```bash
python3 train.py \
  --design balanced_split \
  --init-mode baseline \
  --bias-effective-k 4096 \
  --stage3-epochs 200 \
  --stage3-single-top-k 112 \
  --stage3-pair-top-k 112 \
  --stage3-pair-max-pairs 6216 \
  --population-size 0 \
  --wce-weight 0.05 \
  --score-wce-weight 1.0 \
  --out-dir runs_balanced_split/baseline_w1
```

训练完成后导出 PPL 探针用的 256×256 int16 乘积表（索引 `[a+128, b+128]`，
与 `hard_low_numpy`/RTL 逐位一致）：

```bash
python3 export_int8_lut.py \
  --inits-json runs_balanced_split/baseline_w1/best_signed88_inits.json \
  --out /path/to/LLM-FPGA/outputs/fpga_luts/s88ref_<tag>_signed_int8_lut.npy
```

命名成 `s88ref_<tag>_signed_int8_lut.npy` 放进 `LLM-FPGA/outputs/fpga_luts/`
后，`run_signed_w8a8_ppl_probe.py --designs s88ref_<tag>` 会自动识别。

注意：`train.py --init-mode json` 只读取并校验 INIT，不会自动继承源 artifact 的 calibration、K、loss weights 或 score weights。上例只有在源 JSON 使用默认 CSV、默认权重且 `K=4096` 时才是同目标续跑；否则必须同时显式传回原 `--calibration-csv`、`--calibration-weight-column` 以及所有非默认 loss/score 参数。不要把它和会继承新 artifact 目标的 `refine.py/verify.py` 混淆。

### 14.5 启动32个随机 logit seed

下面示例使用两张 GPU、每张同时一个进程：

```bash
OUT_ROOT="$PWD/runs_random_logits/default_stage3_clean_seed3000_3031" \
NUM_RUNS=32 \
JOBS=2 \
GPU_LIST=0,1 \
SEED_BASE=3000 \
SEED_STRIDE=1 \
K_EFF=4096 \
RANDOM_LOGIT_MEAN=0 \
RANDOM_LOGIT_STD=1 \
POPULATION_SIZE=0 \
./run_default_random_logits.sh \
  --stage3-epochs 100
```

单张 GPU 时，把上面完整命令中的 `JOBS=2`、`GPU_LIST=0,1` 分别替换为 `JOBS=1`、`GPU_LIST=0` 后再执行完整 launcher。

批任务中断后，使用完全相同的参数：

```bash
OUT_ROOT="$PWD/runs_random_logits/default_stage3_clean_seed3000_3031" \
NUM_RUNS=32 JOBS=2 GPU_LIST=0,1 \
SEED_BASE=3000 SEED_STRIDE=1 K_EFF=4096 \
RANDOM_LOGIT_MEAN=0 RANDOM_LOGIT_STD=1 POPULATION_SIZE=0 \
EXISTING_POLICY=resume \
./run_default_random_logits.sh \
  --stage3-epochs 100
```

这里的 resume 是批级续跑：完整 seed 跳过，不完整 seed 先归档再从头训练；没有 optimizer-state 断点续训。

### 14.6 离散 refine

对 CP 设计完整扫描56位 pair：

```bash
python3 refine.py \
  --base-inits-json /path/to/best_signed88_inits.json \
  --out-dir refine/default_pair_new \
  --bit-rounds 20 \
  --pair-rounds 8 \
  --pair-candidate-bits 56 \
  --pair-max-pairs 1540 \
  --basin-iters 0
```

refine 默认继承新 artifact 中的 calibration、K 和 objective weights。复现实验前要确认 `calibration.source` 仍存在；若路径失效，程序会回退到 packaged CSV。

### 14.7 Python与RTL全输入验证

只运行 Python hard 模型和 artifact 检查：

```bash
python3 verify.py \
  --design auto \
  --inits-json /path/to/best_signed88_inits.json \
  --rtl-dir /path/to/best_rtl
```

真正运行 RTL 65,536 输入仿真：

```bash
OSS=/home/xuanqi/tony/work/FPGA/tools/oss-cad-suite

python3 verify.py \
  --design auto \
  --inits-json /path/to/best_signed88_inits.json \
  --rtl-dir /path/to/best_rtl \
  --run-rtl \
  --cells-sim "$OSS/share/yosys/xilinx/cells_sim.v" \
  --iverilog "$OSS/bin/iverilog" \
  --vvp "$OSS/bin/vvp"
```

只出现 `[rtl-artifact] PASS` 代表 metadata 中 INIT 一致，不代表 Verilog 功能仿真完成；必须出现 `[rtl] PASS` 才表示 RTL 对全部65,536输入与 Python hard 模型一致。

---

## 15. 当前 Default 多 seed 实验怎样理解

当前完成的干净32-seed目录：

```text
runs_random_logits/default_stage3_clean_seed2000_2031
```

其结果为：

- 32/32 完成，无失败；
- 产生4个唯一最终 INIT；
- 27个 seed 收敛到同一个最好 INIT；
- 最好 `objective_score=0.0036784894787`；
- `workload_MSE=972.91322`；
- `workload_bias=-0.12374`；
- `workload_ER=42.105%`；
- uniform `WCE=168`。

最好解的 Stage3 路径中：

1. 精确 single 接受1次；
2. 精确 pair 接受1次；
3. Hard-STE block 直接接受0次；
4. 最后完整扫描56个single和1540个pair均无改善；
5. 75/100 epoch 后以 `no_hard_progress` 提前停止。

这说明：

- Stage1/2 梯度训练能把随机垃圾状态拉入稳定 basin；
- 本次最终突破主要来自 Stage3 的精确离散搜索，而不是 Hard-STE 自身翻位；
- 27/32 收敛同解说明这个 basin 很强，但不证明全局最优；
- 最好解只证明当前 Hamming 半径2局部无改进；
- 相对原 RTL/manual baseline，目标明显降低 MSE、bias代理和WCE；相对旧随机Stage2公共终点，WCE保持168，而MSE/bias继续改善。ER允许上升，这是当前权重选择的预期 trade-off；
- 是否改善真实模型 PPL，仍必须由外部 AI 推理测试决定。

其余5个 seed 全部在100 epoch预算的最后一轮仍接受了 pair，随后以 `budget_exhausted` 结束；它们都没有围绕最后的新点再做一次完整无改进扫描，因此不能称为半径2局部最优。

### 15b. balanced_split + WCE 实验（2026-08-22）

针对"default/fast 拓扑天花板"与"WCE 才与 PPL 强相关"两个诊断结论，做了
两项改造并完成一批实验（`runs_balanced_split/`）：

- 拓扑：`balanced_split`（低/中段 INIT 解耦，112 位，资源与 balanced 相同）；
- 目标：`--score-wce-weight 1.0/4.0` + soft `--wce-weight 0.05`，K=4096；
- 训练：7 个 run（baseline/random_logits 起点 × 两档 WCE 权重，Stage3 全预算
  single 112 / pair 6216）+ 一条 `refine.py` 纯离散快速通道。

结果：

- 6/7 个 run 与 refine 通道全部收敛到 uniform `WCE=40` 档（同档内有 3 个
  不同解），远低于手工 balanced 的 80 与 default 训练解的 168；
- 最优解 `wNMSE≈0.000101~0.000118`，比手工 balanced 基线（0.000502）好约
  4~5 倍；多起点一致收敛说明 WCE=40 很可能接近该拓扑在此权重下的结构极限；
- OPT-125M / WikiText2 PPL（seq 512, axcore, per-token/per-channel）：

| 设计 | 4096 tok ΔPPL | 8192 tok ΔPPL | 资源 |
|---|---:|---:|---|
| 手工 balanced | +0.78 | +0.76 | 39 LUT6_2 + 7 CARRY4 |
| 手工 quality | -0.08 | -0.07 | 40 LUT6_2 + 8 CARRY4 |
| `bsplit_base_w1`（4 run 共同解） | +0.40 | +0.43 | 39 LUT6_2 + 7 CARRY4 |
| `bsplit_refine_w1`（refine 通道） | +0.37 | +0.46 | 39 LUT6_2 + 7 CARRY4 |

- 同资源下 PPL 退化相对手工 balanced 约减半；两个最优解差异在探针方差内；
- `rand103_w4` 解 wNMSE/wBias 更好但 PPL (+0.76) 更差，再次证明 workload
  代理与 PPL 非单调，WCE 档位 + 误差结构才是部署侧主导因素；
- 交付物：`runs_balanced_split/wce_batch1/baseline_w1/best_rtl/`（iverilog
  全 65,536 输入 PASS），PPL 探针 LUT 在 `LLM-FPGA/outputs/fpga_luts/
  s88ref_bsplit_base_w1_signed_int8_lut.npy`，报告 `outputs/reports/
  bsplit_final{,_8k}.md`。

---

## 16. 已知边界与下一步方向

### 16.1 当前 objective 是 GEMM 代理，不是 PPL

全局 pair 直方图丢失了 layer、token、输出通道、固定权重位置、量化 scale 和下游敏感度。`K_eff`、global bias 和 conditional bias 比旧 MRED/ER 目标更接近点积误差，但仍不能直接预测 PPL。

最终判断应使用：

```text
exact multiplier through same inference hook
vs.
baseline INIT
vs.
trained/refined candidates
```

并确认不同候选的 INIT 确实加载生效。

### 16.2 训练loss与hard score仍有辅助项错位

主 NMSE/bias 已对齐，但 bit、zero、ER surrogate、MAE 与 hard score 不完全相同。这是 curriculum 与部署选择的刻意分工，也意味着不能只看 soft loss。

### 16.3 STE 是有偏代理

Hard-STE forward 是硬的，backward 却假设阈值导数为1。它适合提供搜索线索，不保证真实 bit flip 改善。当前 exact hard gate 必须保留。

### 16.4 K 与条件偏置需要应用校准

- K 应对应真实 GEMM inner dimension；
- 如果 activation 固定接 A、weight 固定接 B，可以分别观察 A/B conditional RMS；
- 如果网络不同层 K 不同，单个 K 只是折中；
- 更进一步应保存按层/通道分组的 dot-product trace，而不是只有全局 pair histogram。

### 16.5 资源与时序不由训练日志证明

`resource_summary` 是设计声明。训练和 RTL 功能验证不会运行综合，也不会证明最终 LUT/CARRY 被综合器如何裁剪、关键路径多长。最终还需要 Yosys/Vivado 综合与时序报告。

### 16.6 不要在运行中的多seed批次修改代码

不同进程在不同时间启动时，修改 `train.py` 或 imported 模块会造成版本污染。新算法必须使用新的 `OUT_ROOT`；同一批运行期间保持代码和 CSV 不变。

---

## 17. 给老师讲解时的建议顺序

建议按下面八步讲，逻辑最清楚：

1. **硬件目标**：只训练 LUT INIT，不改变拓扑、CARRY4和精确 signed 高位。
2. **统一抽象**：六种 RTL 都实现 `build_model/hard_low_numpy/export_rtl`，所以共用 trainer。
3. **signed 等价式**：`approx=a*b+approx_LL-AL*BL`，说明4096状态只是缓存。
4. **可微化**：INIT bit→logit→sigmoid→soft LUT；Hard阶段用STE穿过阈值。
5. **AI目标**：MSE控制随机误差，平方 bias 控制长点积同向累计，条件 bias 防止分组抵消。
6. **三阶段 curriculum**：Stage1学bit，Stage2转向GEMM，Stage3用hard前向和精确离散邻域落地。
7. **严格验收**：训练loss只负责梯度，最终best必须在离散硬模型上降低 score。
8. **部署验证**：patch 原RTL，Python与RTL全65,536输入验证；最后再做真实PPL和综合时序。

### 常见追问简答

**为什么不用 MRED/ER 当主目标？**  
MRED过度放大小乘积，ER把误差1和误差100都记为一次；MSE和bias更贴近累加数值误差。MRED/ER仍作为很小的诊断项保留。

**为什么 pair 搜索不可少？**  
两个位可能单独翻都让bias变差，一起翻却互相补偿并降低MSE；single贪心无法发现这种协同。

**为什么随机 seed 最后会相同？**  
它们使用同一全批数据、确定性目标和长时间Adam，可能进入同一强吸引域。说明收敛稳定，不说明全局最优。

**为什么不直接相信 Hard-STE 梯度？**  
STE是人为定义的反向近似；真实硬电路不连续，所以候选必须再经过离散 evaluator。

**验证PASS能证明什么？**  
`[rtl] PASS` 证明导出RTL与Python硬模型在全部65,536输入上功能一致；不证明PPL、泛化、综合面积或时序。

**Default与Fast有什么区别？**  
当前算术 mask、资源声明和误差模型相同，区别主要是所复制和patch的 RTL 目录/wrapper。相同 INIT 应产生相同数值行为。

**多seed是否等于更多训练样本？**  
不是。它只增加随机初始化与优化轨迹，所有run仍使用同一校准直方图。

---

## 18. 最终结论

当前训练器本质上是一个混合优化系统：

```text
连续梯度优化负责探索
+
离散hard score负责选择
+
exact single/pair负责修复组合位
+
RTL全输入仿真负责保证部署一致性
```

这比单纯把 LUT INIT 当连续参数训练后直接阈值化更可靠，也比只用 MRED/ER 更贴近 AI GEMM 中的累加误差。它目前优化的是“给定校准分布、K 和代理目标下的可部署 LUT 配置”；真实模型上的最终价值仍以独立 PPL/accuracy 结果为准。

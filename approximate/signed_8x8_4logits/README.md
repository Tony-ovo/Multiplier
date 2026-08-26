本训练目录采用全新的训练思路
我现在想用不一样的思路建模查找表INIT的去训练乘法器，因为目前查找表的每个INIT仅0/1两个值，太过极端，我想把每个INIT的值建模成00/01/10/11，也就是把64位INIT变成128位INIT，然后查找表的输入也要变成12位，一输出的变两输出，两输出的变成四输出。但是最后8*8的16位输出还是要单输出。然后就是需要考虑是面向AI推理训练，需要适合的loss。梯度的计算也要考虑，初步想法是01离00近那就00的梯度占大权重/11的占小权重，然后10的反一下。

---

# quad4：四级 INIT 参数化对照实验（版本 A 落地）

上面是原始想法。落地采用"**四级只放在 INIT 条目、接口不动**"的版本 A 方案：

- **为什么输入/输出不扩位**：训练图里 LUT 之间的信号本来就是连续概率位
  （信息量高于 2-bit），部署时是 1-bit 物理线。把信号扩成 12 输入/4 输出
  会变成 2-bit 数据通路，单个 12 输入函数在 LUT6 架构上要 60+ 个 LUT，
  资源爆炸两个数量级，违背近似乘法器省资源的初衷。
- **四级怎么进训练**：每个可变 INIT 条目 = 4 路 softmax 分类变量
  （00/01/10/11 → 数值 {0, 1/3, 2/3, 1}），前向取期望
  `v = Σ p_k·level_k`。
- **距离加权梯度（原始想法的严格版）**：`∂v/∂logit_k = τ·p_k·(level_k − v)`。
  条目在 00 附近时，11 的 logit 拉力约是 01 的 3 倍（正比于级距），
  10 一侧对称成立——这正是"01 离 00 近则 00 占大权重/11 占小权重"的
  数学形式，且四级全部参与每一步梯度。
- **loss 怎么充分利用四级**：
  1. 主 loss 完全复用 unified trainer 的 GEMM 对齐目标
     （NMSE + 全局/条件 bias、K=4096）+ WCE 平滑极值项，作用在四级期望前向上；
  2. **熵正则**（后期 ramp）：压缩 4 路分布的犹豫度；
  3. **坍缩一致性正则**（后期 ramp）：`(v − round(v))²` 把停在 01/10 的
     条目分别推向 00/11，使最终二值坍缩是小扰动而不是悬崖。
- **hard 阶段（quad-STE）**：条目取 argmax 级 + STE。由于 1/3<0.5<2/3 且
  下游信号在 0.5 处二值决断，**hard 前向逐位等于部署后的二值电路**
  （单测覆盖），反向仍是四级距离加权梯度——四级的作用因此是纯粹的
  梯度重参数化，不引入任何训练-部署 gap。
- **部署坍缩**：{00,01}→0，{10,11}→1，之后用 `refine.py`（同 objective）
  做精确离散抛光兜底。

## 对照设计

唯一变量是参数化（二值 sigmoid logits vs 四级 softmax）：

| | 对照组 | 实验组（本目录） |
|---|---|---|
| 拓扑 | balanced_split（112 位） | 相同 |
| 校准/loss/hard score | GEMM+WCE, K=4096, score-wce 1.0 | 相同（复用同一套代码） |
| 参数化 | 连续 sigmoid logits | **4 路 softmax 四级** |
| 结果位置 | `../signed88_unified_trainer/runs_balanced_split/wce_batch1` | `runs_quad/` |

对照组基线成绩：score 0.002655 / WCE 40 / wNMSE 0.000118（4 个独立起点
收敛的共同解，PPL vs exact +0.40 @4k tokens）。

## 文件

- `quad_lut.py`  QuadLUT6_2：4 路 softmax 条目、距离加权梯度、quad-STE、
  二值坍缩、熵/坍缩正则
- `quad_model.py`  复用 unified 的 balanced_split/CP 核，仅换表类；
  四级使用率统计
- `train_quad.py`  训练器：可选 6000 epoch bit 预热 + Q1 soft 期望 + Q2
  quad-hard STE + population 重启；best 按二值坍缩后的离散 hard objective
  选择，产物与 unified 工具链完全兼容
- `run_quad.sh`  单 run 包装：训练 + refine 离散抛光
- `test_quad.py`  7 项单测（baseline 还原、梯度加权比、hard 前向=部署
  二值、冻结位保护等）

## 用法

```bash
# 单测
python3 -m unittest test_quad -v

# 随机起点（推荐鲁棒配方：6000 epoch bit 预热 + 对齐对照组的 soft ramp）
./run_quad.sh 0 rand100 --design balanced_split --init-mode random_logits \
  --seed 100 --warmup-epochs 6000 --soft-epochs 10000 --hard-epochs 4000 \
  --bit-end 0.05 --mae-start 0.2 --mae-end 0.02 --tau-cycles 1 \
  --population-members 12

# PPL 导出（与 unified 相同）
python3 ../signed88_unified_trainer/export_int8_lut.py \
  --inits-json runs_quad_robust2/bsplit_rand101/refined/best_signed88_inits.json \
  --out ../../LLM-FPGA/outputs/fpga_luts/s88ref_quad_bsplit_rand101_signed_int8_lut.npy
```

## 结果（2026-08-23，7 run × 12000 epoch + refine 离散抛光）

| run | 起点 | train score | refine 后 | WCE | wNMSE | 训练末中间级占比 |
|---|---|---|---|---|---|---|
| **rand104_q** | 随机四级 s104 | 0.029076 | **0.0026408** | 40 | 0.000107 | 9.8% |
| **base_q080** | baseline c0.8 | 0.005740 | **0.0026446** | 40 | 0.000101 | 0% |
| **base_q090** | baseline c0.9 | 0.003370 | **0.0026471** | 40 | 0.000101 | 0% |
| rand102_q | 随机 s102 | 0.035986 | 0.0044393 | 64 | 0.000338 | 16.1% |
| rand100_q | 随机 s100 | 0.011857 | 0.0074249 | 96 | 0.001080 | 10.7% |
| rand103_q | 随机 s103 | 0.021919 | 0.0143527 | 192 | 0.001982 | 19.6% |
| rand101_q | 随机 s101 | 0.041391 | 0.0152500 | 192 | 0.002726 | 17.0% |

对照组（二值 logits，wce_batch1 best）：score **0.0026551** / WCE 40 / wNMSE 0.000118。

关键结论：

1. **前三名超过对照组**（score 低 0.30%~0.54%，wNMSE 低 9%~14%，WCE 持平 40）。
2. **公平性检验通过**：给对照组 best 跑同参数 `refine.py`，结果零改进
   （pair 搜索报 no improvement，score 停在 0.0026551）——对照组已是自身
   盆地的局部最优。quad 的三个解位于**不同且更优的盆地**，改善确实来自
   四级参数化改变的优化路径，而非离散抛光预算差异。
3. 训练过程中随机起点组有 24%~41% 条目真实停留在 01/10 中间级
   （见 `runs_quad/*/quad_stats.json` 轨迹），四级空间被充分利用；
   坍缩正则最终把 baseline 组的中间级全部推平（0%），随机组残余
   9.8%~19.6% 由坍缩 + refine 兜底。
4. 第一版单路径训练下，随机四级起点方差较大（5 个 seed 只有 1 个进入
   最优区）。见下文 **随机鲁棒性 v2**：补上 6000 epoch 纯 bit 预热后
   4/4 随机 seed 全部进入最优区。
5. rand104_q 全输入 65,536 对 numpy 验证 PASS。

## 端到端 PPL 验证（opt-125m / wikitext2 / seq_len 512，ΔPPL vs exact W8A8）

| 设计 | 16k tokens 窗口1 | 16k tokens 窗口2 (offset 16k) |
|---|---|---|
| 对照组 bsplit_base_w1 | +0.360 | +0.470 |
| quad_rand104 | +0.245 | +0.271 |
| **quad_base080** | **+0.014** | **+0.147** |

- 两个独立 16k 窗口排序一致：**quad_base080 < quad_rand104 < 对照组**。
  quad_base080 平均 ΔPPL ≈ +0.08（接近无损），对照组 ≈ +0.42。
- 4k token 的 smoke 窗口噪声约 ±0.3~0.5（不同窗口下排序会洗牌），
  判断这个量级的差异必须用 ≥16k tokens 多窗口口径。
- 报告：`LLM-FPGA/outputs/reports/ppl_quad_batch3_16k.md`、
  `ppl_quad_batch4_16k_off16k.md`；
  LUT：`LLM-FPGA/outputs/fpga_luts/s88ref_quad_{base080,rand104}_signed_int8_lut.npy`。

**当前推荐交付：`runs_quad_robust2/bsplit_rand101/refined/`**（score
0.0026379 / WCE 40，PPL 两窗口 −0.036 / +0.056，同时刷新离散与推理纪录）。
旧推荐 `runs_quad/base_q080/refined/` 仍作对照（PPL +0.014 / +0.147）。

## default 拓扑实验（2026-08-23，7 run 对照：3 对照 + 4 quad）

default 只有 4 个共享表（`cp_lut01/23/45/67`，低/中段共用），56 个可变位。
两组同 objective（GEMM+WCE，K=4096，score-wce 1.0），训练后统一 refine
（56 位全对搜索）。结果（`runs_default_control/`、`runs_quad_default/`）：

| run | refine 后 score | WCE | wNMSE | 与对照 baseline 同解? |
|---|---|---|---|---|
| control baseline / rand100 / rand104 | 0.0139324 | 168 | 0.0019631 | — / 是 / 是 |
| quad base_q080 / base_q090 | 0.0139324 | 168 | 0.0019631 | 是 / 是 |
| quad rand100_q | 0.0622485 | 525 | 0.0170642 | 否 |
| quad rand104_q | 0.1024518 | 714 | 0.0325009 | 否 |

结论：

1. **5 个独立 run（两种参数化 × 多起点，对照组含 population 多重启）
   逐位收敛到同一个 INIT 解**——该解几乎可以确定就是 default 拓扑
   56 位空间的全局最优（与历史无-WCE 口径最优解的 WCE/wNMSE 完全一致，
   加 WCE 目标与四级参数化都无法再改进）。
2. default 拓扑天花板 vs balanced_split：WCE 168 vs 40（4.2×），
   wNMSE 0.00196 vs 0.000101（19×）。**共享表结构是硬约束**，
   这正是 balanced_split 解开共享的价值所在。
3. quad 参数化在 default 上与二值参数化殊途同归（好解空间太小，
   不存在"更好的盆地"可找）。第一版单路径训练下随机四级起点更难收敛；
   见下文 **随机鲁棒性 v2**，补 bit 预热后 3/3 随机 seed 全部收敛到
   同一全局最优。
4. 该解全输入 65,536 对验证 PASS；
   LUT：`LLM-FPGA/outputs/fpga_luts/s88ref_default_best_signed_int8_lut.npy`。
5. **PPL（16k 双窗口，ΔPPL vs exact W8A8）：default 最优解 +9.38 / +7.40，
   在 AI 推理上不可用**；同条件 balanced_split quad_base080 为
   +0.014 / +0.147（接近无损）。即使把 default 训到拓扑极限，
   其精度也远不满足推理需求——结构升级（balanced_split）是必要的，
   不是训练方法能弥补的。
   报告：`ppl_default_16k.md`、`ppl_default_16k_off16k.md`。

## 随机鲁棒性 v2（2026-08-24）

第一版随机起点失败的根因不是四级参数化，而是训练器缺了对照组的
**6000 epoch 纯 bit 预热**（逐位 BCE 把 LUT 结构从噪声里拉出来）。
v1 只加了 population 重启 + 温度重加热：有改善但未进最优区，且温度
cycle 反而让主训练变差。v2 配方对齐对照组：warmup 6000 + soft 10000
+ hard 4000 + 12 个奇偶强弱扰动成员，`--tau-cycles 1`。

结果（`runs_quad_robust2/`，7/7 随机 seed 全部达标）：

| run | train score | refine 后 | WCE | wNMSE | 与历史最优同解? |
|---|---|---|---|---|---|
| **bsplit_rand101** | 0.0036334 | **0.0026379** | 40 | 0.000103 | 新解（刷新纪录） |
| bsplit_rand103 | 0.0037599 | 0.0026434 | 40 | 0.000107 | 新解 |
| bsplit_rand100 | 0.0040236 | 0.0026446 | 40 | 0.000101 | 同 base_q080 |
| bsplit_rand200 | 0.0032173 | 0.0026551 | 40 | 0.000118 | 同对照组 wce_batch1 |
| default_rand100 / 104 / 200 | 0.060 / 0.016 / 0.060 | **0.0139324** | 168 | 0.001963 | 全部同 default 全局最优 |

结论：

1. **随机鲁棒性达标**：balanced_split 4/4、default 3/3 全部进入各自
   拓扑的最优区。default 三个随机 seed 逐位收敛到同一 INIT。
2. **best 出自主训练而非 population**（stage 均为 `quad_soft` /
   `quad_hard_ste`）——bit 预热才是把随机起点送进正确盆地的关键，
   population 是保险而非主因。
3. `bsplit_rand101` 刷新 balanced_split 历史最优：score **0.0026379**
   （此前 rand104_q 0.0026408 / base_q080 0.0026446），全输入 65,536
   对验证 PASS。LUT：`s88ref_quad_bsplit_rand101_signed_int8_lut.npy`。
4. 四个 bsplit 随机 seed 落到四个不同盆地，全部 WCE=40、score 在
   0.00264±0.00002——最优区是一片相邻盆地，不是单点。
5. **PPL（opt-125m / wikitext2 / seq_len 512，16k 双窗口，ΔPPL vs exact
   W8A8）**：四个 bsplit 随机 seed 离散分数几乎一样（0.00264±0.00002），
   推理差距却很大——**离散 score 不能单独当 PPL 代理**。

   | run | INIT | ΔPPL 窗口1 / 窗口2 |
   |---|---|---|
   | **bsplit_rand101** | 独特 | **−0.036 / +0.056** |
   | bsplit_rand100 | 同 base_q080 | +0.014 / +0.147 |
   | bsplit_rand200 | 同二值对照 | +0.360 / +0.470 |
   | bsplit_rand103 | 独特 | +0.752 / +0.580 |
   | default 三个 | 同 default 全局最优 | +9.38 / +7.40 |

   报告：`ppl_quad_robust2_16k.md`、`ppl_quad_rand103_16k.md` 及对应 `_off16k`。

## default_split（2026-08-25）

Default 的 6×6 把 `b` 拆成三个 2-bit 数字，RTL 里本来就有三个
`s8862_approx62_cp` 实例，只是 INIT 被绑成同一套。`default_split` 把
lo/mid/hi 解开成 12 张独立表（168 可变位），**资源仍是 37 LUT6_2 + 6 CARRY4**。
基线（INITs 绑定时）与 Default 完全一致：MAE=23.625、WCE=336。

训练配方对齐 v2（warmup 6000 + soft 10000 + hard 4000 + 12 pop），
结果写在 `runs_default_split/`（12 run 全部完成）。

| run | refine score | WCE | 备注 |
|---|---|---|---|
| **rand102** | **0.012709** | **160** | 独特解；首次打破共享表 WCE=168 |
| rand200 | 0.013361 | 168 | |
| rand104 | 0.013376 | 168 | |
| base_q055/070/080 | 0.013429 | 168 | 三个 baseline 逐位同解 |
| rand101 | 0.013454 | 168 | |
| base_q090 | 0.013467 | 168 | |
| rand105 | 0.013487 | 168 | |
| rand100 | 0.013517 | 168 | |
| rand201 | 0.059478 | 608 | 未进最优区 |
| rand103 | 0.074887 | 512 | 未进最优区 |

对照共享表 default 天花板 0.013932 / WCE 168。rand102 全输入 65,536 验证 PASS；
相对共享表 score 低 **8.8%**，WCE 168→160。其余成功 seed 仍锁在 WCE 168，
改善约 3%~4%。2/8 随机 seed 未收敛——default_split 比共享表有空间，但盆地更碎。

```bash
OUT_BASE=runs_default_split PAIR_BITS=168 PAIR_MAX=14028 ./run_quad.sh 0 base080 \
  --design default_split --init-mode baseline --init-conf 0.8 --seed 1 \
  --warmup-epochs 6000 --soft-epochs 10000 --hard-epochs 4000 \
  --bit-end 0.05 --mae-start 0.2 --mae-end 0.02 --tau-cycles 1 \
  --population-members 12
```

# approx_66_mlp62 训练说明

这个目录是在训练一个 `6*6` 近似乘法器。核心思路不是训练一个普通 MLP 去做乘法，而是把 RTL 里的 LUT INIT 当成可训练参数：Python 训练脚本用可微的 soft LUT 模拟 Verilog 结构，训练结束后把每个 LUT 的概率值二值化，导出真正能写回 RTL 的 `64'h...` INIT。

当前自动流程是 1 到 5 步串起来跑：

```text
step1 shared random search
  -> step2 unshared fullcomp
  -> step3 unshared paircomp
  -> step4 escape local search
  -> step5 conservative WCE search
```

每个 restart 是一条独立搜索链。`run_1to5_parallel.py` 可以同时启动多条链，但同一条链内部必须按 1 到 5 顺序跑，因为后一步要吃前一步的 `best_approx66_inits.json`。

## 乘法器结构

输入是：

```text
a[5:0], b[5:0]
```

训练脚本把 `b` 拆成三个 2-bit 段：

```text
low  = b[1:0]
mid  = b[3:2]
high = b[5:4]
```

然后分别做三个 `6*2` 近似乘法：

```text
plow  = approx62(a, b[1:0])
pmid  = approx62(a, b[3:2])
phigh = approx62(a, b[5:4])
```

每个 `approx62` 输出 8 bit：

```text
p0, p1, p2, p3, p4, p5, p6, p7
```

最后 `comp66` 把 `plow/pmid/phigh` 拼成 12-bit 的 `6*6` 近似乘积。

在最终 paircomp 结构里，大致是：

```text
prod[0], prod[1]   = plow[0], plow[1]
prod[10], prod[11] = phigh[6], phigh[7]

prod[2], prod[3] = u_comp23(plow[2], pmid[0], plow[3], pmid[1])
prod[4]          = u_comp4 (plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1])
prod[5]          = u_comp5 (plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1])
prod[6]          = u_comp6 (plow[6], pmid[4], phigh[2], plow[7], pmid[5], phigh[3])
prod[7]          = u_comp7 (plow[6], pmid[4], phigh[2], plow[7], pmid[5], phigh[3])
prod[8], prod[9] = u_comp89(pmid[6], phigh[4], pmid[7], phigh[5])
```

所以最终主要训练这些 INIT：

```text
low_lut1,  low_lut2,  low_lut3,  low_lut4
mid_lut1,  mid_lut2,  mid_lut3,  mid_lut4
high_lut1, high_lut2, high_lut3, high_lut4
u_comp23, u_comp89
u_comp4, u_comp5, u_comp6, u_comp7
```

## 训练到底在训练什么

每个 LUT 的 `INIT[63:0]` 在 RTL 里本来是 0/1。训练时不能直接对 0/1 求梯度，所以脚本给每个 INIT bit 建一个连续的 `logit`。

流程可以理解成：

```text
logit
  -> sigmoid 得到 0 到 1 的概率
  -> sharp01 把概率推向 0/1
  -> soft LUT 查表
  -> 输出再 sharp01
  -> STE 二值化
  -> 拼成 approx product
  -> 计算 loss
  -> 反向传播更新 logit
```

二值化用的是 STE：

```python
hard = (x >= 0.5).to(x.dtype)
y = hard.detach() - x.detach() + x
```

前向传播时，`y` 看起来是硬的 0/1；反向传播时，梯度近似当作从连续的 `x` 传回去。所以梯度不是从真正离散的 0/1 bit 里产生的，而是通过二值化前的连续概率/logit 近似传回去。

这也是为什么 INIT 初始化影响很大：训练过程能走梯度，但最后评价的是硬二值 INIT。两个连续概率很接近的模型，二值化以后可能落到完全不同的离散解。


## Step 1: shared random search

相关文件：

```text
1/train_approx66_ste_search.py
```

step1 是最早、最小的模型。它只训练：

```text
lut1, lut2, lut3, lut4, u_or23, u_or89
```

其中 `lut1..lut4` 是 shared 的，也就是说 low/mid/high 三个 `approx62` 共用同一套 6*2 LUT INIT。

step1 的 comp66 结构里：

```text
prod[2], prod[3] = u_or23 训练
prod[4], prod[5], prod[6], prod[7] = 固定 OR3
prod[8], prod[9] = u_or89 训练
```

自动脚本里 step1 又分两段：

```text
01_explore
02_refine
```

### 01_explore

从随机 INIT 开始，探索大方向：

```text
--init-mode random
--random-init-prob 默认 0.5
--epochs 默认 4000
--lr 0.01
--c-anneal
--noise-std 0.25
```

这里比较像“撒网”。它不保证每次都好，主要目标是找到一个还不错的 basin。

### 02_refine

从 `01_explore/best_approx66_inits.json` 继续细搜：

```text
--init-mode json
--lr 0.002
--epochs 默认 1000
--noise-std 0.05
--restart-from-best-every 120
--bitflip-after
--bitflip-rounds 3
```

这一段会不断用当前 best 重新初始化并加小扰动，最后再做少量 greedy bit-flip。

step1 输出：

```text
01_step1_shared_random/final_best_approx66_inits.json
01_step1_shared_random/final_best_approx66_verilog_snippet.v
```

## Step 2: unshared fullcomp

相关文件：

```text
2/train_approx66_unshared_fullcomp.py
2/run_approx66_unshared_fullcomp.sh
```

step2 做两个重要扩展。

第一，把 shared 的 `lut1..lut4` 展开成三套：

```text
low_lut1..low_lut4
mid_lut1..mid_lut4
high_lut1..high_lut4
```

这样 low/mid/high 三个 `approx62` 可以学出不同的 INIT。

第二，把 step1 里固定的 `prod[4:7]` OR3 改成可训练 LUT6：

```text
u_or4, u_or5, u_or6, u_or7
```

这一步叫 `fullcomp`，因为 comp66 中间列的逻辑开始从固定 OR 变成可训练 LUT。

脚本分三段：

```text
01_expand
02_refine
03_bitflip_best
```

### 01_expand

从 step1 JSON 出发，把 shared INIT 自动复制成 low/mid/high 三套，然后用小学习率训练：

```text
--epochs 900
--lr 0.001
--init-p 0.92
--noise-std 0.03
--bitflip-rounds 20
```

### 02_refine

继续小步细搜：

```text
--epochs 900
--lr 0.0006
--init-p 0.95
--noise-std 0.015
--bitflip-rounds 20
```

### 03_bitflip_best

只做离散 bit-flip，不再做梯度训练：

```text
--bitflip-only
--bitflip-rounds 60
--bitflip-mode best
```

bit-flip 的意思是：直接在已经二值化的 LUT INIT 里翻某一个 bit。

例如某个 LUT 原来是：

```text
u_or4 INIT bit[17] = 0
```

翻转以后变成：

```text
u_or4 INIT bit[17] = 1
```

或者反过来从 1 变成 0。每翻一个 bit，脚本都会用硬件等价的 hard evaluator 重新枚举全部 `64*64=4096` 个输入，重新计算 MRED/MED/ER/WCE。

它不是梯度下降，而是很直接的离散搜索：

```text
当前 best INIT
  -> 尝试翻一个候选 bit
  -> 重新计算硬 MRED
  -> 如果 MRED 更低，就保留这个翻转
  -> 否则撤销
```

`best` 模式会扫描候选 bit，选择当前能让 MRED 降最多的 bit 翻转，比 first-improvement 慢，但更稳。

`first` 和 `best` 的区别：

```text
first:
  按顺序扫描 bit，遇到第一个能改善 MRED 的翻转就接受。
  优点是快。
  缺点是可能错过本轮更好的翻转。

best:
  扫完所有候选 bit，选择改善最大的那个翻转。
  优点是每一轮更贪心、更稳。
  缺点是慢很多。
```

为什么 step2 后面要做 bitflip：STE 训练结束后，logit 已经被二值化成真实 INIT，但这个二值 INIT 不一定是附近最优。bitflip 相当于在当前解附近做“单 bit 精修”，把梯度训练没对齐好的个别 INIT bit 修正掉。

step2 输出：

```text
02_step2_unshared_fullcomp/final_best_approx66_inits.json
02_step2_unshared_fullcomp/final_best_approx66_unshared_fullcomp.v
```

## Step 3: unshared paircomp

相关文件：

```text
3/train_approx66_unshared_paircomp.py
3/run_approx66_unshared_paircomp_next.sh
```

step3 继续改 comp66 结构。step2 的：

```text
u_or4, u_or5, u_or6, u_or7
```

在 step3 变成：

```text
u_comp4, u_comp5, u_comp6, u_comp7
```

区别是 pair-aware。

step2 中 `prod[4]` 和 `prod[5]` 分别只看自己的三个输入：

```text
prod[4] 看 plow[4], pmid[2], phigh[0]
prod[5] 看 plow[5], pmid[3], phigh[1]
```

step3 中 `u_comp4` 和 `u_comp5` 都同时看两列的信息：

```text
plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1]
```

同理，`u_comp6/u_comp7` 同时看 prod[6]/prod[7] 两列的信息。

这样多了一些局部补偿能力。代价是搜索空间更大，INIT 更依赖初始点。

脚本分四段：

```text
01_paircomp_train
02_target_pair_flip
03_basin_hop
04_final_single_polish
```

### 01_paircomp_train

从 step2 best JSON 出发，扩展成 pair-aware comp 结构，然后继续 STE 训练：

```text
--epochs 800
--lr 0.0008
--single-after
--single-rounds 20
--single-mode best
```

### 02_target_pair_flip

这一步不做梯度训练，只做 pair bit-flip，也就是一次同时翻两个 INIT bit。

pair-flip 可以理解成 bit-flip 的二阶版本。

普通 bit-flip 一次只翻一个 bit：

```text
flip A
```

pair-flip 一次同时翻两个 bit：

```text
flip A + flip B
```

它存在的原因是：有些改进不是单独翻一个 bit 能得到的。可能出现这种情况：

```text
只翻 A: MRED 变差，不接受
只翻 B: MRED 变差，不接受
同时翻 A+B: MRED 变好，应该接受
```

如果只做 single-bit 搜索，这种解永远找不到，因为第一步翻 A 或翻 B 都会被拒绝。pair-flip 就是专门用来跳过这种“单 bit 局部最优”的小障碍。

搜索逻辑是：

```text
当前 best INIT
  -> 选两个候选 bit
  -> 同时翻转
  -> 枚举 4096 个输入重新计算硬指标
  -> 如果满足条件并且 MRED 更低，就保留
  -> 否则撤销
```

默认重点搜：

```text
u_comp4,u_comp5,u_comp6,u_comp7,
u_comp23,u_comp89,
low_lut3,mid_lut3,high_lut3,
low_lut2,mid_lut2,high_lut2
```

因为这些 LUT 对中间列和 MRED 影响通常更大。

pair-flip 的组合数量增长很快。如果有 `N` 个候选 bit，理论 pair 数是：

```text
N * (N - 1) / 2
```

所以脚本不会默认全量扫所有 LUT 的所有 bit，而是用 `--pair-lut-names` 和 `--pair-max-pairs` 限制范围。当前这个阶段叫 `target_pair_flip`，意思就是“只优先扫最可能有收益的重点 LUT”。

### 03_basin_hop

这是为了跳出 single-bit local optimum。

逻辑是：

```text
从当前 best 出发
随机翻 2 到 5 个 bit
做 single-bit 局部搜索
如果最终 MRED 更低，就接受
重复多次
```

basin 可以理解成一个局部最优区域。当前 INIT 周围如果所有 single-bit flip 都不能改善，甚至很多 pair-flip 也不能改善，那么这个点就是一个局部坑。继续在附近一小步一小步走，很可能一直回到同一个解。

basin hopping 的做法更粗暴一点：

```text
1. 从当前 best 或 beam 里的一个较好解出发。
2. 随机选择 k 个 bit，一次性翻转，k 通常是 2 到 5。
3. 这个新点通常会变差，但它可能已经跳到了另一个 basin 附近。
4. 对这个新点做 single-bit local polish，把它往附近局部最优收敛。
5. 如果 polish 后比原来的 best 更好，就接受成新 best。
6. 如果没有更好，就丢掉这次扰动，继续下一次。
```

所以 basin hopping 不是“随机乱翻然后直接保存”。它的关键是：

```text
随机多 bit 扰动 + 局部精修 + 只接受更好结果
```

这和普通 bitflip/pairflip 的关系是：

```text
bitflip:
  在当前点附近走 1 bit。

pairflip:
  在当前点附近走 2 bit。

basin_hop:
  先跳远一点，再做局部搜索，看能不能落到更好的坑里。
```

这一步对 LUT INIT 搜索很重要，因为当前问题的离散空间很崎岖。很多时候单 bit 和 pair bit 已经找不到路了，但多翻几个“看起来暂时变差”的 bit，再局部修回来，可能得到更低的 MRED。

### 04_final_single_polish

最后再做一轮全局 single-bit best-improvement 收尾：

```text
--single-only
--single-rounds 30
--single-mode best
--single-lut-names all
```

step3 输出：

```text
03_step3_paircomp/final_best_approx66_inits.json
03_step3_paircomp/final_best_approx66_unshared_paircomp.v
```

## Step 4: escape local search

相关文件：

```text
4/train_approx66_escape_local.py
4/run_approx66_escape_local_next.sh
```

step4 主要不是训练 logits，而是在二值化后的 INIT 空间里做更强的离散搜索。它的目标是：当前结果如果已经是 single-bit 或 pair-bit 局部最优，就通过多 bit 扰动跳出去。

脚本分四段：

```text
01_broad_escape
02_comp_lut23_escape
03_segment_escape
04_final_single
```

### 01_broad_escape

在全部有效 bit 上做比较宽的 escape：

```text
--pool-lut-names all
--iters 80
--neutral-top 220
--kmin 2
--kmax 8
```

它会找一些“单独翻转影响不大”的 neutral bits，然后随机组合翻转，再做局部 polish。

### 02_comp_lut23_escape

把搜索重点放到 comp 和 lut2/lut3：

```text
u_comp4,u_comp5,u_comp6,u_comp7,
u_comp23,u_comp89,
low_lut2,low_lut3,
mid_lut2,mid_lut3,
high_lut2,high_lut3
```

这些位置通常更容易影响中间列误差。

### 03_segment_escape

只围绕 low/mid/high 三个 approx62 段做 escape。MRED 对小 exact 的 case 很敏感，所以 low 段的扰动有时会带来明显变化。

### 04_final_single

最后使用 paircomp evaluator 做全局 single-bit best-improvement：

```text
--single-only
--single-rounds 80
--single-mode best
```

step4 输出：

```text
04_step4_escape_local/final_best_approx66_inits.json
04_step4_escape_local/final_best_approx66_unshared_paircomp.v
```

## Step 5: conservative WCE search

相关文件：

```text
5/train_approx66_conservative_wce.py
5/run_approx66_conservative_wce_next.sh
```

step5 是保守搜索。它不是单纯追最低 MRED，而是在限制 WCE 的条件下继续找更低 MRED。

它会输出两类结果：

```text
strict:  WCE <= 930
relaxed: WCE <= 1000
```

脚本分三段：

```text
01_top_guided_wce930
02_neutral_escape_wce930
03_relaxed_wce1000_candidate
```

### 01_top_guided_wce930

先找相对误差最大的 case，再根据这些 case 反推出可能相关的 LUT bit，做 constrained single/pair search。

约束：

```text
--max-wce 930
```

也就是说，只有翻转后仍然满足 `WCE <= 930`，并且 MRED 更低，才会接受。

### 02_neutral_escape_wce930

继续在 `WCE <= 930` 约束下做 neutral multi-bit escape：

```text
--escape-iters 160
--kmin 1
--kmax 5
--escape-pair-after
```

这个阶段是 strict WCE 结果的主要来源。

### 03_relaxed_wce1000_candidate

稍微放宽 WCE：

```text
--max-wce 1000
```

这个结果不会覆盖 strict best，而是单独保存成 relaxed candidate。因为有时候稍微放宽 WCE 能换来更低的 MRED，后面做 8*8 组合时可以一起评估。

step5 输出：

```text
05_step5_conservative_wce/final_wce930_best_approx66_inits.json
05_step5_conservative_wce/final_wce930_best_approx66_unshared_paircomp.v

05_step5_conservative_wce/final_wce1000_candidate_approx66_inits.json
05_step5_conservative_wce/final_wce1000_candidate_approx66_unshared_paircomp.v
```

## 如何自动跑 1 到 5

最简单：

```bash
cd /home/xuanqi/tony/work/FPGA/approx_mlp/approx_66_mlp62
./run.sh
```

`run.sh` 默认参数：

```text
GPU_LIST=0
NUM_RUNS=8
JOBS=2
OUT_ROOT=runs_1to5_8runs
```

等价于：

```bash
./run_1to5_parallel.py \
  --num-runs 8 \
  --jobs 2 \
  --seed-base 0 \
  --seed-stride 1000 \
  --cuda-devices 0 \
  --out-root runs_1to5_8runs
```

含义：

```text
--num-runs 8
  一共跑 8 条独立随机初始化链。

--jobs 2
  同时跑 2 条链。

--cuda-devices 0
  都放到 GPU 0。

--cuda-devices 0,1
  多条 restart 轮流分配到 GPU 0 和 GPU 1。

--seed-base 0 --seed-stride 1000
  restart_00 seed=0
  restart_01 seed=1000
  restart_02 seed=2000
```

你也可以这样指定：

```bash
GPU_LIST=0 NUM_RUNS=16 JOBS=4 OUT_ROOT=runs_1to5_16runs ./run.sh
```

断点续跑：

```bash
./run_1to5_parallel.py \
  --out-root runs_1to5_8runs \
  --num-runs 8 \
  --jobs 2 \
  --cuda-devices 0 \
  --resume
```

`--resume` 的逻辑是：如果某一步已经有 final 输出，就跳过那一步；没有 final 输出的步骤会重新跑。

## 输出目录怎么看

一次自动 run 的结构类似：

```text
runs_1to5_8runs/
  summary.json
  overall_best_wce930_approx66_inits.json
  overall_best_wce930_approx66_unshared_paircomp.v
  overall_best_wce1000_approx66_inits.json
  overall_best_wce1000_approx66_unshared_paircomp.v

  restart_00_seed_0/
    status.json
    pipeline.log
    01_step1_shared_random/
    02_step2_unshared_fullcomp/
    03_step3_paircomp/
    04_step4_escape_local/
    05_step5_conservative_wce/

  restart_01_seed_1000/
    ...
```

重点看：

```text
summary.json
  总结所有 restart 的 strict/relaxed 最好结果。

restart_xx_seed_xxxx/status.json
  这条链跑到了哪一步，最终指标是多少。

restart_xx_seed_xxxx/pipeline.log
  这条链从 step1 到 step5 的完整终端日志。

overall_best_wce930_approx66_inits.json
  所有 restart 里 WCE<=930 的最好结果。

overall_best_wce1000_approx66_inits.json
  所有 restart 里 relaxed WCE<=1000 的最好结果。
```

注意：有些子脚本会在阶段结束时才复制 final JSON。训练正在跑时，`terminal_log.txt` 里的当前 best 可能比你打开的 `best_approx66_inits.json` 更新；最终比较最好等脚本跑完，看 final/overall 文件。

## 为什么随机 INIT 影响很大

这个问题很正常，而且是这个任务的核心难点。

原因是：

```text
1. INIT 是离散 0/1 空间，搜索空间巨大。
2. STE 只是给二值化提供近似梯度，不保证全局最优。
3. step1 如果进入差的 basin，后面 step2-step5 多数只是局部改良。
4. MRED 对小 exact 特别敏感，低位 LUT 的少量错误可能放大成很大的相对误差。
5. single/pair/escape 都有局部性，不能保证从任意随机 seed 都救回来。
```

所以你之前手动一个一个跑，有时比自动 1-5 好，是因为你手动做了“筛选”：看到哪个阶段结果好，就把它作为下一步输入。当前自动脚本每个 restart 是单线传递，没有在每一阶段做全局 top-K 筛选。

更理想的自动策略是 beam search：

```text
step1 跑很多随机 INIT
选 top-K 个进入 step2
step2 再选 top-K 个进入 step3
step3 再选 top-K 个进入 step4
step4/5 再做局部搜索
```

这样比单线 restart 更接近你之前手动挑选的过程。

## 当前比较结果的注意事项

历史旧 JSON 的 MRED 不一定是当前 tb 对齐口径。比较历史结果时，最好用当前 evaluator 重新计算，或者用：

```text
MRED_new = MRED_old * 3969 / 4096
```

例如目前已知历史结果里：

```text
5/runs_conser_0/final_wce1000_candidate_approx66_inits.json
  旧口径 MRED = 0.0893541388
  新口径 MRED ~= 0.086584

5/runs_conser_0/final_wce930_best_approx66_inits.json
  旧口径 MRED = 0.0952012938
  新口径 MRED ~= 0.092249

5/runs_conser_3/final_wce930_best_approx66_inits.json
  旧口径 MRED = 0.1030050823
  新口径 MRED ~= 0.099811
```

如果当前某个新 run 的 MRED 是 `0.12` 左右，那它按统一口径并不如这些历史强结果。

## 实验建议

短期建议：

```text
1. 继续让当前 1-5 跑完。
2. 最终只比较 final/overall 文件，不要比较中途 JSON。
3. 比较历史结果时统一用当前 evaluator 重算。
4. 保留 strict WCE<=930 和 relaxed WCE<=1000 两类候选。
```

中期建议：

```text
1. 把历史 best INIT 作为 elite seed 加进自动流程。
2. 不要每次完全从随机开始。
3. 增加每阶段 top-K 选择，而不是每个 restart 单线跑到底。
4. 保存 Pareto 候选：best MRED, best WCE<=930, best WCE<=1000, low MED, low ER。
```

长期建议：

```text
把 run_1to5_parallel.py 升级成 beam pipeline。
```

当前的 `run_1to5_parallel.py` 适合批量跑随机 restart；下一版如果要更稳，就应该让每一阶段都横向比较多个候选，再把最好的 K 个送入下一阶段。这会比单纯增加 epoch 更有效。

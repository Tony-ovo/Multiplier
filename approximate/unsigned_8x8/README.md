# approx_88_mlp 总览

这个目录用于训练和评估 8x8 近似乘法器。现在已经按实验路线整理成 `1/2/3/4` 四个子目录，顶层不再保留旧的 `run.sh`、`eval_cross_modes.py`、`cross_mode_eval.json`。

当前归档位置：

```text
3/run.sh
3/eval_cross_modes.py
3/cross_mode_eval.json
```

原因是这三个文件都服务于 cross62/hybrid 路线，放在 `3/` 更合适。

## 8x8 分解

输入拆成高 2 bit 和低 6 bit：

```text
a = {ah, al}, ah = a[7:6], al = a[5:0]
b = {bh, bl}, bh = b[7:6], bl = b[5:0]
```

8x8 乘法写成：

```text
a*b = al*bl + ((bl*ah + al*bh) << 6) + ((ah*bh) << 12)
```

对应模块：

```text
LL = approx66(al, bl)
HL = 6x2 block for bl*ah
LH = 6x2 block for al*bh
HH = ah*bh
```

训练路线的差异主要就在：

- `LL` 是否训练。
- `HL/LH` 用 accurate62 还是 approx62。
- 顶层 `comp88` 是否也纳入可训练 LUT。

## 指标口径

当前 8x8 默认 MRED 和 `tb_88.v` 对齐：

```text
err  = abs(approx - exact)
mask = exact > 0
MRED = sum(err[mask] / exact[mask]) / 65536
```

也就是相对误差只对 `exact != 0` 累加，但最后除以全部 `256*256=65536` 个输入组合。

## 目录 1：只训练 low66

说明文档：

```text
1/README.md
```

核心结构：

```text
LL = trainable approx66
HL/LH = accurate62
HH = exact
comp88 = exact/fixed
```

这是精度优先路线。只近似低位 6x6，交叉项仍保持准确。

当前最好结果：

```text
1/runs88_04/final_best_approx88_inits.json
MRED = 0.021524731481489378
WCE  = 914
```

适合：

- 想要更好的 MRED/WCE。
- 可以接受 HL/LH 使用 accurate62 的资源开销。

## 目录 2：继承 6x6，训练 HL/LH approx62

说明文档：

```text
2/README.md
```

核心结构：

```text
LL = inherited/trainable approx66
HL/LH = trainable approx62
HH = exact
comp88 = exact/fixed
```

这是早期 cross62 路线，目标是减少 HL/LH 的资源。

当前最好结果：

```text
2/runs88_cross_03/final_best_approx88_cross62_inits.json
MRED = 0.02827479174100164
WCE  = 2048
```

适合：

- 对比 cross62 的早期结果。
- 看从 `projected` 初始化一路训练的效果。

## 目录 3：cross62 多路线和 hybrid 评估

说明文档：

```text
3/README.md
```

核心结构同目录 2：

```text
LL = inherited/trainable approx66
HL/LH = trainable approx62
HH = exact
comp88 = exact/fixed
```

目录 3 现在放了 cross62 的主要工具：

```text
3/run.sh
3/eval_cross_modes.py
3/cross_mode_eval.json
```

`3/run.sh` 是旧的兼容入口，支持：

```text
MODE=low66
MODE=cross62
MODE=eval_modes
```

不过新实验建议优先直接使用各目录自己的脚本。

`3/eval_cross_modes.py` 用来评估 HL/LH 的不同组合：

```text
aa: HL accurate62, LH accurate62
ap: HL accurate62, LH fixed approx62
pa: HL fixed approx62, LH accurate62
pp: HL fixed approx62, LH fixed approx62
jj: HL/LH 使用 JSON 里训练得到的 approx62
```

运行示例：

```bash
cd /home/xuanqi/tony/work/FPGA/approx_mlp/approx_88_mlp/3

MODE=eval_modes \
BASE_JSON=best_approx66_inits.json \
OUT_JSON=cross_mode_eval.json \
./run.sh
```

已有 `cross_mode_eval.json` 的关键结果：

```text
aa: MRED=0.0148978620, WCE=914,  cross ~= 12 LUT6_2 + 2 CARRY4
pp: MRED=0.0266462752, WCE=4034, cross ~=  8 LUT6_2 + 0 CARRY4
```

目录 3 当前两条路线：

```text
runs88_cross_01~04              CROSS_INIT_MODE=projected
runs88_cross_pp/all_mred_s*     CROSS_INIT_MODE=approx62
```

当前最好结果：

```text
3/runs88_cross_03/final_best_approx88_cross62_inits.json
MRED = 0.026426319952271
WCE  = 2284
```

适合：

- 研究 resource-first 的 pure approx62 cross 路线。
- 做 accurate/approx/hybrid cross 的资源和误差对比。

## 目录 4：8x8 cascade 联合训练

说明文档：

```text
4/README.md
```

核心结构：

```text
LL = inherited/trainable approx66
HL/LH = trainable approx62
HH = exact
comp88 = trainable LUT cascade
```

这是当前更接近你设想的版本：不只训练 low66 和 HL/LH，还把顶层 `comp88` 的 LUT INIT 也纳入训练。

新增可训练顶层 LUT：

```text
u88_gp0..u88_gp8
```

多 seed 两 GPU/单 GPU 启动脚本：

```text
4/run_multi_seed.sh
```

只放到 GPU1 并一次性启动所有 seed：

```bash
cd /home/xuanqi/tony/work/FPGA/approx_mlp/approx_88_mlp/4

SEEDS="500 800 1100 1400 1700 2000" \
GPU_LIST="1" \
CROSS_INIT_MODE=approx62 \
TRAIN_MAX_WCE=-1 \
MAX_WCE=4500 \
ESCAPE_ITERS=40 \
OUT_PREFIX=runs88_cascade_gpu1 \
./run_multi_seed.sh
```

每个 seed 的日志在自己的目录：

```text
4/runs88_cascade_gpu1_s500/pipeline.log
```

适合：

- 继续冲更好的 MRED。
- 让 8x8 整体更像“级联 LUT/MLP 网络”一起训练。
- 尝试让顶层 comp88 学会补偿 HL/LH approx62 的误差。

## 当前路线对比

| 目录 | 主要近似位置 | 当前最好 MRED | WCE | 资源倾向 |
|---|---|---:|---:|---|
| `1` | 只近似 LL low66 | 0.0215247315 | 914 | 资源较多，精度最好 |
| `2` | LL + HL/LH approx62 | 0.0282747917 | 2048 | 更省资源 |
| `3` | LL + HL/LH approx62，多初始化/评估 | 0.0264263200 | 2284 | 更省资源 |
| `4` | LL + HL/LH + comp88 顶层 LUT | 正在实验 | 待定 | 更激进，可调空间最大 |

## 文件归档说明

原来顶层的三个文件已经移动：

```text
旧: eval_cross_modes.py      -> 新: 3/eval_cross_modes.py
旧: cross_mode_eval.json     -> 新: 3/cross_mode_eval.json
旧: run.sh                   -> 新: 3/run.sh
```

移动后不影响 `1/2/3/4` 各自脚本：

- `1` 仍然用 `1/run_approx88_*.sh`。
- `2` 仍然用 `2/run_approx88_cross62_*.sh`。
- `3` 仍然用 `3/run_approx88_cross62_*.sh`，也可以用 `3/run.sh` 做兼容入口。
- `4` 仍然用 `4/run_multi_seed.sh` 或 `4/run_approx88_cascade_pipeline.sh`。

## 建议怎么继续

如果目标是论文/对比结果：

- 保留 `1` 作为精度优先 baseline。
- 保留 `3` 作为 pure approx62 cross 的资源优先 baseline。
- 用 `3/eval_cross_modes.py` 做 `aa/ap/pa/pp/jj` 的资源和误差表。

如果目标是继续优化：

- 重点跑 `4`。
- 多试 `CROSS_INIT_MODE=approx62` 和 `CROSS_INIT_MODE=projected`。
- 多换 `BASE_JSON`，不要只对同一个 basin 重复多 seed。



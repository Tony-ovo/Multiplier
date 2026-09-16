# BalancedSplit

Balanced 拓扑的 INIT 解耦版本。算术结构与 `Balanced`（`APPROX_MASK=3'b011`）
完全一致：`b[1:0]`、`b[3:2]` 两个 digit 段使用 carry-predicted 近似 6x2，
`b[5:4]` 段与全部 signed 高位修正保持精确。

与 `Balanced` 的唯一差别：两个近似段不再共用一个 `s8862_approx62_cp`
模块，而是各自独立的 `s8862_approx62_cp_lo` / `s8862_approx62_cp_mid`。
两个实例本来就物理存在，因此资源不变（39 LUT6_2 + 7 CARRY4），
但可训练自由度从 56 位翻倍到 112 位：权重为 x1 的低段和权重为 x4 的
中段可以学习不同的近似/误差抵消策略。

| 文件 | 内容 |
|---|---|
| `s8862_approx62_cp_split.v` | lo/mid 两个独立 INIT 的近似 6x2 |
| `s8862_approx66_split.v` | lo/mid 近似 + high 精确 + 压缩器 |
| `s8862_mul62.v` | 精确 6x2（高段使用，冻结） |
| `s8862_comp66_q6.v` | 精确压缩器（冻结） |
| `s8862_fused_mac.v` | signed 高位 fused MAC（冻结） |
| `signed88_approx_balanced_split.v` | 顶层 + signed finish |
| `signed88_approx_top.v` | 标准 `s88_top` 包装 |

基线 INIT 与原 CP 相同，因此 baseline 指标与 Balanced 完全一致
（uniform：ER≈19.1%、MAE≈5.625、WCE=80、bias=0）。

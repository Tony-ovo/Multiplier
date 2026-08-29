# DefaultSplit

Default/Fast 拓扑的 INIT 解耦版本。算术结构与 `Default`（`APPROX_MASK=3'b111`）
完全一致：`b[1:0]`、`b[3:2]`、`b[5:4]` 三个 digit 段全部使用
carry-predicted 近似 6x2，全部 signed 高位修正保持精确。

与 `Default` 的唯一差别：三个近似段不再共用一个 `s8862_approx62_cp`
模块，而是各自独立的 `s8862_approx62_cp_lo` / `_mid` / `_hi`。
三个实例本来就物理存在，因此资源不变（37 LUT6_2 + 6 CARRY4），
但可训练自由度从 56 位变为 168 位：权重 x1 / x4 / x16 的三段可以学习
不同的近似与误差抵消策略。

| 文件 | 内容 |
|---|---|
| `s8862_approx62_cp_dsplit.v` | lo/mid/hi 三个独立 INIT 的近似 6x2 |
| `s8862_approx66_dsplit.v` | 三段近似 + 压缩器 |
| `s8862_comp66_q6.v` | 精确压缩器（冻结） |
| `s8862_fused_mac.v` | signed 高位 fused MAC（冻结） |
| `signed88_approx_default_split.v` | 顶层 + signed finish |
| `signed88_approx_top.v` | 标准 `s88_top` 包装 |

基线 INIT 与原 CP 相同，因此 baseline 指标与 Default 完全一致
（uniform：MAE=23.625、WCE=336、bias=0）。

# Aggressive

统一综合顶层：`s88_top`

内部实现：`signed88_approx_aggressive`

把本目录下全部 `.v` 文件加入工程即可。LL 完全由 LUT 构成，不使用
`CARRY4`；完整乘法器中的四个 `CARRY4` 全部来自两个精确 fused MAC。

```text
31 LUT6_2 + 4 LUT6 + 4 CARRY4
MAE=98.413086, WCE=898, bias=-5.492188
```

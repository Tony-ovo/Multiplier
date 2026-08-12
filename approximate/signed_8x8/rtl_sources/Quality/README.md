# Quality

统一综合顶层：`s88_top`

内部实现：`signed88_approx_quality`

把本目录下全部 `.v` 文件加入工程即可。本版本只近似最低权重的 6×2 子块，
其余两个子块保持精确。

```text
40 LUT6_2 + 8 CARRY4
MAE=1.125, WCE=16, bias=0
```

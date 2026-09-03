# Balanced

统一综合顶层：`s88_top`

内部实现：`signed88_approx_balanced`

把本目录下全部 `.v` 文件加入工程即可。本版本近似低、中的两个 6×2 子块，
最高权重子块保持精确。

```text
39 LUT6_2 + 7 CARRY4
MAE=5.625, WCE=80, bias=0
```

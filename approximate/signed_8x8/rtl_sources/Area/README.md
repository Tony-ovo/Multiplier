# Area

统一综合顶层：`s88_top`

内部实现：`signed88_approx_area`

把本目录下全部 `.v` 文件加入工程即可。本版本将 AL 量化到 16 的倍数，并
切断 LL 压缩器的低进位段，以 LUT 数量最少为目标。

```text
29 LUT6_2 + 5 CARRY4
MAE=128.104492, WCE=634, bias=+11.75
```

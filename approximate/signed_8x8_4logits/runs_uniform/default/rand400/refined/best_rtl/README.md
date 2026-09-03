# Default

统一综合顶层：`s88_top`

内部实现：`signed88_approx`

把本目录下全部 `.v` 文件加入工程即可。`signed88_approx` 是 Fast 版本的默认
封装，展平后的资源和误差与 Fast 完全相同。

```text
37 LUT6_2 + 6 CARRY4
MAE=23.625, WCE=336, bias=0
```

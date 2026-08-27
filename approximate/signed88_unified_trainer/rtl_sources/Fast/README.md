# Fast

统一综合顶层：`s88_top`

内部实现：`signed88_approx_fast`

把本目录下全部 `.v` 文件加入工程即可。三个 6×2 子块全部采用无 CARRY4
的局部进位预测实现。

```text
37 LUT6_2 + 6 CARRY4
MAE=23.625, WCE=336, bias=0
```

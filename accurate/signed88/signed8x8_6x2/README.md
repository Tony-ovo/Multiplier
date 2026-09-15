# signed88_6x2

这是一个精确的二进制补码有符号 8×8 组合乘法器。LL 的无符号 6×6
乘法器使用三个 6×2 子乘法器和一个压缩器实现，后半部分沿用两级 fused
有符号 digit 累加结构。

## 顶层接口

```verilog
module signed88_6x2 (
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output wire signed [15:0] prod
);
```

设计为纯组合逻辑，只显式例化 `LUT6_2` 和 `CARRY4`。

## 文件

| 文件 | 内容 |
|---|---|
| `signed88_6x2.v` | 顶层、两个共享 LUT 和两级 fused MAC 连接 |
| `s8862_mul62.v` | 精确无符号 6×2，5 LUT6_2 + 1 CARRY4 |
| `s8862_mul66.v` | 三个 6×2 组成的精确无符号 6×6 |
| `s8862_comp66.v` | 三个移位 6×2 结果的压缩器 |
| `s8862_fused_mac.v` | 两个高十位融合累加级 |
| `tb_signed88_6x2.v` | 65,536 组输入穷举测试平台 |

## 总体分解

```text
A = AL + 64*AH
B = BL + 64*BH

A*B = AL*BL + 64*(AH*BL + A*BH)
```

其中 `AL=a[5:0]`、`BL=b[5:0]` 为无符号 6 位数；`AH=a[7:6]`、
`BH=b[7:6]` 按二位补码解释。

令：

```text
LL = AL*BL = L0 + 64*H0
H1 = H0 + AH*BL       (mod 2^10)
H2 = H1 + BH*A        (mod 2^10)
prod = {H2,L0}
```

## 三个 6×2 组成 LL

低 6×6 使用：

```text
P0 = AL * BL[1:0]
P1 = AL * BL[3:2]
P2 = AL * BL[5:4]

LL = P0 + (P1 << 2) + (P2 << 4)
```

三个 `s8862_mul62` 各用 5 个 `LUT6_2` 和一个 `CARRY4`。压缩器使用
9 个 `LUT6_2` 和两个 `CARRY4`。

每个 6×2 的最高部分积由父模块提供：

```text
ll_high_1 = AL[5] & BL[1]
ll_high_2 = AL[5] & BL[3]
ll_high_3 = AL[5] & BL[5]
```

前两个最高位共享一颗 LUT：

```text
INIT = 64'ha0a0a0a088888888
O5   = ll_high_1
O6   = ll_high_2
```

第二颗共享 LUT 同时产生第三个最高位和第一级 fused MAC 的 `q6`：

```text
INIT = 64'hf300f30088888888
O5   = ll_high_3 = AL[5] & BL[5]
O6   = stage1_q6 = a[7] & (a[6] | ~BL[5])
```

因此 LL 的完整资源（包括三个最高位和 `q6` 的父层逻辑）为：

```text
3 × s8862_mul62       15 LUT6_2 + 3 CARRY4
s8862_comp66           9 LUT6_2 + 2 CARRY4
两个共享 LUT           2 LUT6_2
------------------------------------------------
LL 相关合计            26 LUT6_2 + 5 CARRY4
```

## fused MAC

二位有符号 digit `y` 表示 `y[0]-2*y[1]`：

| `y[1:0]` | 数值 | 加法行 |
|---:|---:|---|
| `00` | 0 | 0 |
| `01` | +1 | `x` |
| `10` | -2 | `~(x<<1)+1` |
| `11` | -1 | `~x+1` |

通用融合 LUT 同时产生：

```text
O5 = q
O6 = q XOR acc
INIT = 64'hd973268c268c268c
```

二者分别连接 `CARRY4.DI` 与 `CARRY4.S`，`CYINIT=y[1]` 注入负行所需的
补码 `+1`。部分积生成和累加传播因此共用同一层 LUT。

两个 MAC 的资源为：

```text
stage1 = 7 LUT6_2 + 2 CARRY4
stage2 = 9 LUT6_2 + 2 CARRY4
```

## 最终资源

Yosys 展平和 `synth_xilinx -family xc7` 得到一致统计：

```text
Number of cells: 51
  LUT6_2    42
  CARRY4     9
```

没有 `LUT6`、DSP、寄存器、RAM、隐式乘法器或通用加法单元。

与四个 3×3 的 fused43 版本相比：

| 版本 | LUT | CARRY4 |
|---|---:|---:|
| 四个 3×3 LL | 43 | 6 |
| 三个 6×2 LL | **42** | **9** |

本版本少一个物理 LUT，但多三个 `CARRY4`，属于以更多进位链换取更少 LUT
的资源点。

## 验证

使用 Xilinx 原语仿真模型遍历了全部 `256×256=65,536` 组输入：

```text
PASS: all 65536 signed 8x8 input combinations are correct.
```

在 FPGA 工程根目录可复现：

```bash
tools/oss-cad-suite/bin/iverilog -g2012 \
  -s tb_signed88_6x2 -o /tmp/signed88_6x2_sim \
  /usr/share/yosys/xilinx/cells_sim.v \
  signed8x8_6x2/s8862_mul62.v \
  signed8x8_6x2/s8862_comp66.v \
  signed8x8_6x2/s8862_mul66.v \
  signed8x8_6x2/s8862_fused_mac.v \
  signed8x8_6x2/signed88_6x2.v \
  signed8x8_6x2/tb_signed88_6x2.v

tools/oss-cad-suite/bin/vvp /tmp/signed88_6x2_sim
```

综合时将 `signed88_6x2` 设置为顶层，测试平台不要加入 synthesis sources。


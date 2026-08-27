# Dot-product grouped trace 格式草案（v1）

旧的 `a,b,count/p_calib` CSV 仍由 `signed88.data.load_calibration_csv()` 原样支持。新格式用于保留一次真实 GEMM/dot-product 内部的误差累加关系，供 application-aware 训练和 validation hard 选优使用。

## 为什么只需 4096 个状态

当前六类 signed8x8 RTL 都只近似 `AL*BL`，其中：

```text
AL = a & 63
BL = b & 63
state = AL*64 + BL                    # 0..4095
e[state] = approximate_LL - exact_LL  # approx - exact
```

因此，一个 dot group 的反量化输出误差可以精确写成：

```text
delta_y[g] = scale[g] * sum_state(count[g,state] * e[state])
```

`count` 保留同一个 dot 内误差同向累加或相互抵消的信息；平坦的 `(a,b)` 直方图不能保留这一信息。如果同一 dot 中不同 MAC 使用不同乘积 scale，应按 scale 拆成多个 group，不能把它们直接合并。

## JSONL

第一行必须是：

```json
{"type":"metadata","format":"signed88-dot-trace-v1","state_count":4096,"error_convention":"approx_minus_exact"}
```

此后每行表示一个 dot/group：

```json
{"type":"group","id":"token17.ffn.down.42","layer":"model.layers.3.mlp.down_proj","channel":"42","split":"train","scale":0.00006103515625,"sensitivity":1.72,"normalizer":0.125,"counts":[[0,3],[65,7],[4095,1]]}
```

字段语义：

- `id`：全文件唯一的 group 标识。
- `layer`、`channel`：用于分别约束 layer bias 和 `(layer,channel)` bias；channel 名可以复用，但会和 layer 联合分组。
- `split`：只能是 `train`、`validation` 或 `test`。训练搜索只用 train，hard INIT 选择必须用 validation，test 只做最终报告。
- `scale`：整数乘积误差到该 dot 输出域的 groupwise scale，必须有限且非零。
- `sensitivity`：该输出对下游 NLL/PPL 的正权重要度，必须严格大于 0。建议用 held-out 精确模型得到的梯度平方/Fisher 对角近似，而不是与候选 INIT 一起更新；不需要的 group 应直接省略，而不是写成 0 权重。
- `normalizer`：可选、默认 1；用于让不同层的 `delta_y` 可比较，例如该层精确输出 RMS、量化步长或离线确定的容差，必须为正。
- `counts`：非空稀疏 `[state,count]` 列表。state 必须是 0..4095 的唯一整数，count 必须是正整数。

不要让 validation/test token 出现在 train 中；否则 hard score 仍然是 in-sample 指标。

## 代码接口

```python
from signed88.dot_trace import (
    DotProxyLossConfig, compute_dot_proxy_loss, evaluate_dot_trace,
    load_dot_trace_jsonl, make_dot_trace_task_evaluator, to_torch_dot_trace,
)

profile = load_dot_trace_jsonl('dot_trace.jsonl')
train_profile = profile.select_split('train')
validation_profile = profile.select_split('validation')
train_batch = to_torch_dot_trace(train_profile, device)

# low_value 是模型的 4096 项近似 LL，grid_exact_ll 是精确 LL。
error_table = low_value - model.grid_exact_ll
loss, terms = compute_dot_proxy_loss(
    error_table, train_batch, DotProxyLossConfig()
)

# 每次候选 INIT 二值化以后，在 validation 上进行纯 NumPy hard 选优。
hard_error = design.hard_low_numpy(inits) - exact_ll
metrics = evaluate_dot_trace(hard_error, validation_profile)
print(metrics.proxy_score, metrics.weighted_mae, metrics.channel_bias_rms)

# 可直接传给 selection_v2.HardwareSelector(task_evaluator=...)。
task_evaluator = make_dot_trace_task_evaluator(profile, split='validation')
```

默认 proxy 由 sensitivity-weighted Huber、layer/channel 条件 bias、尾部 CVaR 和可选 MSE 组成。它是 PPL 风险的代理而不是 PPL 数值预测；最终仍应以独立 token 上的真实 NLL/PPL 和网络精度验收。

原 CSV 和 grouped trace 的自动兼容入口是：

```python
profile = load_objective_profile(path)  # .csv 或 .jsonl/.ndjson
```

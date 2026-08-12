# Unified signed INT8 approximate-multiplier trainer

This project is generated from the supplied RTL directories (`Aggressive`, `Default`, `Fast`, `Balanced`, `Quality`, `Area`) and the supplied W8A8 signed joint calibration histogram.

## 1. What is trained

The **training inputs are signed INT8 pairs** `(a,b)` from the calibration CSV and the principal losses are computed from the **final signed INT16 product**. For all supplied RTLs the signed upper correction is exact and only the `AL*BL` low 6x6 block is approximate, so

```text
approx_signed(a,b) = a*b + approx_LL(AL,BL) - AL*BL
AL = a & 63, BL = b & 63
```

The implementation evaluates the 4096 possible `(AL,BL)` states once per epoch and gathers them for all signed CSV rows. This is only an exact cache optimization: the workload MRED/ER/MAE/bias losses are still evaluated row-by-row using the signed 8-bit operands and signed 16-bit exact products.

Only Design-declared LUT INIT bits are PyTorch parameters. Fixed carry chains, exact children, signed fused MAC stages, and fixed compressor LUTs are not optimized.

## 2. One trainer, multiple RTL structures

The trainer/refiner never branches on `aggressive/fast/balanced/...`. Hardware-specific behavior lives behind the Design API.

```text
train.py / refine.py / verify.py
             |
             v
        BaseDesign API
             |
   +---------+---------+---------+
   |         |         |         |
Aggressive  CP-Hybrid  Area    future design plugins
             |
      Fast/Balanced/Quality
```

Supported names:

- `aggressive`: 18 trainable truth tables, matching the unshared aggressive RTL.
- `fast`: three CP 6x2 children approximate; four CP INIT tables are shared as in the supplied module definition.
- `default`: same arithmetic configuration as Fast, but exports/patches the supplied `Default/` RTL package and its `signed88_approx` wrapper.
- `balanced`: low and middle CP children approximate; high child exact.
- `quality`: low CP child approximate; middle/high children exact.
- `area`: two q16 INIT tables trainable; the low-carry-cut compressor remains fixed.

For CP designs, digits 0/1/2 remain exact: only reachable INIT addresses for digit 3 are mutable. This preserves the supplied carry-prediction architecture rather than silently changing its contract.

## 3. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run regressions first:

```bash
python -m unittest discover -s tests -v
```

The baseline tests must reproduce the RTL README metrics:

| design | uniform MAE | WCE | bias |
|---|---:|---:|---:|
| aggressive | 98.4130859375 | 898 | -5.4921875 |
| fast/default | 23.625 | 336 | 0 |
| balanced | 5.625 | 80 | 0 |
| quality | 1.125 | 16 | 0 |
| area | 128.1044921875 | 634 | +11.75 |

## 4. Train one design

Fast:

```bash
python train.py \
  --design fast \
  --calibration-csv data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv \
  --out-dir runs/fast_seed0 \
  --seed 0
```

Balanced:

```bash
python train.py --design balanced --out-dir runs/balanced_seed0 --seed 0
```

Aggressive:

```bash
python train.py --design aggressive --out-dir runs/aggressive_seed0 --seed 0
```

Use the supplied baseline instead of random INIT initialization:

```bash
python train.py --design quality --init-mode baseline --out-dir runs/quality_baseline_tune
```

Resume/tune a tagged result:

```bash
python train.py \
  --design fast \
  --init-mode json \
  --base-inits-json runs/fast_seed0/best_signed88_inits.json \
  --out-dir runs/fast_retune
```

The default curriculum is:

1. soft auxiliary low-product bit supervision + signed-output MAE;
2. soft ramp from bit supervision toward **signed-output MRED**;
3. hard STE LUT forward focused on signed-output error;
4. optional small INIT perturbation + local population retraining.

The best candidate is selected by a hard discrete score, not by the differentiable surrogate loss:

```text
1.00 * workload_MRED
+0.25 * workload_ER
+0.10 * workload_NED
+0.05 * |workload_bias| / 16384
+0.05 * uniform_MRED
```

## 5. Multi-seed runs

```bash
DESIGN=fast NUM_RUNS=8 JOBS=2 GPU_LIST=0,1 ./run_multi.sh
```

Extra `train.py` arguments can be passed with `EXTRA_ARGS`, for example:

```bash
DESIGN=balanced NUM_RUNS=4 EXTRA_ARGS='--population-size 0 --stage1-epochs 2000' ./run_multi.sh
```

## 6. Exact hard refinement

```bash
python refine.py \
  --base-inits-json runs/fast_seed0/best_signed88_inits.json \
  --out-dir refine/fast_seed0
```

`refine.py` is topology-independent. It asks the Design for its searchable INIT bits and hard vector evaluator, then performs single-bit, pair-bit, and small basin-hop search using the real signed workload objective.

## 7. Verify

Python hard evaluation plus explicit 65,536 signed pairs:

```bash
python verify.py \
  --inits-json refine/fast_seed0/best_signed88_inits.json \
  --rtl-dir refine/fast_seed0/best_rtl
```

If Icarus Verilog and Xilinx `cells_sim.v` are available:

```bash
python verify.py \
  --inits-json refine/fast_seed0/best_signed88_inits.json \
  --rtl-dir refine/fast_seed0/best_rtl \
  --run-rtl \
  --cells-sim /path/to/xilinx/cells_sim.v
```

The generated RTL is not a handwritten reimplementation. `export_rtl()` copies the supplied RTL template directory and patches only the Design-declared `.INIT(...)` bindings. A `trained_artifact.json` is written beside the RTL.

## 8. Full train -> top-k -> refine pipeline

```bash
DESIGN=fast NUM_RUNS=8 TOP_K=2 ./run_pipeline.sh
```

## 9. Adding a new hardware structure

Create a new Design plugin under `signed88/hardware/designs/` and implement:

```python
class MyDesign(BaseDesign):
    spec = DesignSpec(...)

    def build_core(self, inits, init_conf, noise_std):
        ...  # differentiable PyTorch low-core model

    def hard_low_numpy(self, inits):
        ...  # exact 4096-state hard evaluator
```

Then register it in `signed88/hardware/registry.py`.

The trainer, loss code, refiner, metric evaluator, run scripts, and verifier do not need modification. `DesignSpec` also declares the baseline INITs, mutable/search bit policy, RTL directory, and exact RTL `.INIT` binding locations.

If a future multiplier no longer has the common “exact signed wrapper + approximate low 6x6” structure, add another model wrapper implementing the same signed `forward` contract; do not add design-name branches to the trainer.

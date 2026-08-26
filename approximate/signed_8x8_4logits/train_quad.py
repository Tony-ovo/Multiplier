#!/usr/bin/env python3
"""Train a signed 8x8 approximate multiplier with 4-level (2-bit) INIT entries.

Controlled experiment against the binary-logit unified trainer: same topology
(balanced_split by default), same calibration data, same GEMM-aligned loss,
same discrete hard objective (including the uniform-WCE term) and the same
artifact format -- only the INIT parameterization differs (QuadLUT6_2).

Phases:
  Q1 soft  : expected-value forward over the 4-level softmax, temperature and
             GEMM ramps; late in the phase an entropy term plus a collapse
             regularizer start pushing 01->00 and 10->11.
  Q2 quad-hard : argmax-level STE forward (numerically identical to the
             deployed binary circuit because 1/3<0.5<2/3) with 4-level
             distance-weighted gradients.

Best checkpoints are always selected on the discrete binary-collapsed hard
objective via signed88.metrics.evaluate_design, so results are directly
comparable with runs_balanced_split/wce_batch1.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

_UNIFIED = Path(__file__).resolve().parents[1] / 'signed88_unified_trainer'
if str(_UNIFIED) not in sys.path:
    sys.path.insert(0, str(_UNIFIED))

from signed88.common import (  # noqa: E402
    GEMM_OBJECTIVE_SCHEMA, ObjectiveWeights, set_seed, validate_objective_weights, write_json,
)
from signed88.data import load_calibration_csv, to_torch  # noqa: E402
from signed88.hardware import get_design  # noqa: E402
from signed88.losses import LossConfig, compute_loss  # noqa: E402
from signed88.metrics import evaluate_design  # noqa: E402

from quad_model import (  # noqa: E402
    build_quad_model, collapse_regularizer, quad_usage_stats, randomize_quad_logits,
    reset_quad_from_inits,
)

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', default='balanced_split')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=str(ROOT / 'runs_quad/run0'))
    p.add_argument('--device', default='auto')
    p.add_argument('--calibration-csv', default=str(_UNIFIED / 'data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv'))
    p.add_argument('--calibration-weight-column', default='auto',
                   choices=['auto', 'count', 'p_calib', 'weight', 'probability'])

    p.add_argument('--init-mode', default='baseline', choices=['baseline', 'random_logits'])
    p.add_argument('--init-conf', type=float, default=0.999,
                   help='softmax mass on the extreme level for baseline entries')
    p.add_argument('--init-noise-std', type=float, default=0.0)
    p.add_argument('--random-logit-mean', type=float, default=0.0)
    p.add_argument('--random-logit-std', type=float, default=1.0)

    p.add_argument('--tau0', type=float, default=3.0, help='base softmax temperature')
    p.add_argument('--tau-start', type=float, default=0.5, help='c_init factor at soft start')
    p.add_argument('--tau-end', type=float, default=2.0, help='c_init factor at soft end / hard phase')
    p.add_argument('--tau-cycles', type=int, default=1,
                   help='number of low->high temperature warm-restart cycles in the soft phase '
                        '(>1 helps random starts explore before committing)')

    # Population-based restarts from the running best (mirrors the unified
    # trainer mechanism that gives binary-logit runs their random-start
    # robustness).  Each member re-seeds the quad logits from the current best
    # binary solution at low confidence plus noise, then trains a short
    # soft+hard cycle; improvements update the global best for later members.
    p.add_argument('--population-members', type=int, default=0)
    p.add_argument('--population-soft-epochs', type=int, default=1200)
    p.add_argument('--population-hard-epochs', type=int, default=600)
    p.add_argument('--population-init-conf', type=float, default=0.53)
    p.add_argument('--population-noise-std', type=float, default=0.5)
    p.add_argument('--population-strong-conf', type=float, default=0.51,
                   help='odd members restart with this (weaker) confidence...')
    p.add_argument('--population-strong-noise-std', type=float, default=1.0,
                   help='...and this larger noise, to escape poor basins')
    p.add_argument('--population-lr', type=float, default=None,
                   help='defaults to --lr')

    p.add_argument('--warmup-epochs', type=int, default=0,
                   help='constant-weight bit-supervision warmup before the soft ramp '
                        '(bit=warmup-bit, gemm=gemm-start, tau fixed at tau-start); '
                        'mirrors unified stage1 and is essential for random starts')
    p.add_argument('--warmup-bit', type=float, default=1.0)
    p.add_argument('--warmup-mae', type=float, default=0.25)
    p.add_argument('--soft-epochs', type=int, default=8000)
    p.add_argument('--hard-epochs', type=int, default=4000)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--hard-lr-scale', type=float, default=0.3)
    p.add_argument('--grad-clip', type=float, default=1.0)

    # Quad-specific regularizers ("loss uses all four levels").
    p.add_argument('--collapse-weight', type=float, default=0.5,
                   help='final weight of the (v - round(v))^2 collapse consistency term')
    p.add_argument('--collapse-ramp-start', type=float, default=0.6,
                   help='fraction of soft epochs after which the collapse ramp begins')
    p.add_argument('--entropy-weight', type=float, default=0.05,
                   help='final weight of the categorical entropy term (bin_reg)')

    # GEMM loss terms: identical defaults to the unified trainer + wce.
    p.add_argument('--calibration-mix', type=float, default=0.98)
    p.add_argument('--mse-weight', type=float, default=1.0)
    p.add_argument('--bias-weight', type=float, default=1.0)
    p.add_argument('--conditional-bias-weight', type=float, default=0.001)
    p.add_argument('--bias-effective-k', type=float, default=4096.0)
    p.add_argument('--er-weight', type=float, default=0.0001)
    p.add_argument('--zero-weight', type=float, default=0.25)
    p.add_argument('--bit-weighting', default='linear', choices=['uniform', 'linear', 'sqrt_value', 'value'])
    p.add_argument('--wce-weight', type=float, default=0.05)
    p.add_argument('--wce-beta', type=float, default=0.25)
    p.add_argument('--er-temperature-start', type=float, default=4.0)
    p.add_argument('--er-temperature-end', type=float, default=0.10)

    # Aux term ramps (soft phase), mirroring stage1->stage2 of the unified flow.
    p.add_argument('--bit-start', type=float, default=1.0)
    p.add_argument('--bit-end', type=float, default=0.2)
    p.add_argument('--mae-start', type=float, default=0.5)
    p.add_argument('--mae-end', type=float, default=0.0)
    p.add_argument('--mred-start', type=float, default=0.2)
    p.add_argument('--mred-end', type=float, default=0.05)
    p.add_argument('--gemm-start', type=float, default=0.02)
    p.add_argument('--gemm-end', type=float, default=1.0)
    p.add_argument('--gemm-ramp-frac', type=float, default=0.4)

    # Hard objective weights (checkpoint selection), matching wce_batch1.
    p.add_argument('--score-mse-weight', type=float, default=1.0)
    p.add_argument('--score-mred-weight', type=float, default=0.0001)
    p.add_argument('--score-er-weight', type=float, default=0.00005)
    p.add_argument('--score-ned-weight', type=float, default=0.05)
    p.add_argument('--score-bias-weight', type=float, default=1.0)
    p.add_argument('--score-conditional-bias-weight', type=float, default=0.001)
    p.add_argument('--score-uniform-mred-weight', type=float, default=0.0001)
    p.add_argument('--score-wce-weight', type=float, default=1.0)

    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--print-every', type=int, default=200)
    return p.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(spec)


def lerp(a: float, b: float, t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return a + (b - a) * t


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    design = get_design(args.design)
    profile = load_calibration_csv(Path(args.calibration_csv), args.calibration_weight_column)
    batch = to_torch(profile, device)

    objective = ObjectiveWeights(
        workload_mred=args.score_mred_weight,
        workload_er=args.score_er_weight,
        workload_ned=args.score_ned_weight,
        workload_bias=args.score_bias_weight,
        uniform_mred=args.score_uniform_mred_weight,
        workload_nmse=args.score_mse_weight,
        workload_conditional_bias=args.score_conditional_bias_weight,
        bias_effective_k=args.bias_effective_k,
        uniform_wce=args.score_wce_weight,
    )
    validate_objective_weights(objective)
    loss_cfg = LossConfig(
        calibration_mix=args.calibration_mix,
        er_weight=args.er_weight,
        mse_weight=args.mse_weight,
        bias_weight=args.bias_weight,
        conditional_bias_weight=args.conditional_bias_weight,
        bias_effective_k=args.bias_effective_k,
        zero_weight=args.zero_weight,
        symmetry_weight=0.0,
        bin_weight=0.0,  # entropy weight is scheduled manually per epoch
        bit_weighting=args.bit_weighting,
        wce_weight=args.wce_weight,
        wce_beta=args.wce_beta,
    )

    model = build_quad_model(design, design.spec.base_inits, args.init_conf, args.init_noise_std, args.tau0).to(device)
    init_provenance = {'mode': args.init_mode}
    if args.init_mode == 'random_logits':
        gen = torch.Generator(device='cpu')
        gen.manual_seed(args.seed)
        # Draw on CPU for cross-device reproducibility, then copy over.
        cpu_model = build_quad_model(design, design.spec.base_inits, args.init_conf, 0.0, args.tau0)
        init_provenance.update(randomize_quad_logits(cpu_model, args.random_logit_mean, args.random_logit_std, gen))
        model.load_state_dict(cpu_model.state_dict())
        model = model.to(device)

    def collapsed_inits() -> dict:
        return design.normalize_inits(model.hard_inits())

    def evaluate_now() -> tuple:
        inits = collapsed_inits()
        return inits, evaluate_design(design, inits, profile, objective)

    calibration_meta = profile.metadata()
    artifact_common = {
        'objective_schema': GEMM_OBJECTIVE_SCHEMA,
        'objective_weights': objective.__dict__,
        'calibration': calibration_meta,
        'train_args': vars(args),
        'parameterization': 'quad4_softmax',
    }

    initial_inits, initial_metrics = evaluate_now()
    write_json(out / 'initial_signed88_inits.json', design.artifact(
        initial_inits, metrics=initial_metrics.to_dict(),
        extra={**artifact_common, 'stage': 'initial', 'quad_usage': quad_usage_stats(model), 'init_provenance': init_provenance}))
    best = {'stage': 'initial', 'epoch': -1, 'inits': initial_inits, 'metrics': initial_metrics}
    print(f'[design] {design.spec.name} quad4 entries={quad_usage_stats(model)["mutable_entries"]}')
    print(f'[device] {device.type}')
    print(f'[initial] {initial_metrics.short()}')

    history = (out / 'history.jsonl').open('w', encoding='utf-8')
    usage_track = []
    global_epoch = 0

    def save_best(stage: str, epoch: int, inits: dict, metrics) -> None:
        nonlocal best
        best = {'stage': stage, 'epoch': epoch, 'inits': inits, 'metrics': metrics}
        write_json(out / 'best_signed88_inits.json', design.artifact(
            inits, metrics=metrics.to_dict(),
            extra={**artifact_common, 'stage': stage, 'best_epoch': epoch,
                   'quad_usage': quad_usage_stats(model)}))

    def run_phase(stage: str, epochs: int, hard_middle: bool, lr: float,
                  warmup: bool = False) -> None:
        nonlocal global_epoch
        if epochs <= 0:
            return
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for local in range(epochs):
            t = local / max(epochs - 1, 1)
            if warmup:
                # Constant-weight structure learning: strong per-bit BCE, GEMM
                # barely on, low fixed temperature, no quad regularizers yet.
                c_init = args.tau_start
                bit_w, mae_w = args.warmup_bit, args.warmup_mae
                mred_w, gemm_w = 0.0, args.gemm_start
                collapse_w = entropy_w = 0.0
                er_tau = args.er_temperature_start
            elif hard_middle:
                c_init = args.tau_end
                bit_w, mae_w, mred_w, gemm_w = args.bit_end, args.mae_end, args.mred_end, args.gemm_end
                collapse_w = args.collapse_weight
                entropy_w = args.entropy_weight
                er_tau = args.er_temperature_end
            else:
                # Temperature may run several low->high warm-restart cycles;
                # every other schedule (aux weights, regularizer ramps) stays
                # on global phase progress so late-phase commitment still holds.
                if args.tau_cycles > 1 and t < 1.0 - 1e-9:
                    ct = t * args.tau_cycles
                    ct = ct - math.floor(ct)
                else:
                    ct = 1.0 if t >= 1.0 - 1e-9 else t
                c_init = lerp(args.tau_start, args.tau_end, ct)
                ramp_t = min(t / max(args.gemm_ramp_frac, 1e-6), 1.0)
                bit_w = lerp(args.bit_start, args.bit_end, t)
                mae_w = lerp(args.mae_start, args.mae_end, t)
                mred_w = lerp(args.mred_start, args.mred_end, t)
                gemm_w = lerp(args.gemm_start, args.gemm_end, ramp_t)
                late = max(0.0, (t - args.collapse_ramp_start) / max(1.0 - args.collapse_ramp_start, 1e-6))
                collapse_w = args.collapse_weight * late
                entropy_w = args.entropy_weight * late
                er_tau = math.exp(lerp(math.log(args.er_temperature_start), math.log(args.er_temperature_end), t))
            opt.zero_grad(set_to_none=True)
            total, terms = compute_loss(
                model, batch, c_init=c_init, c_out=1.0, hard_middle=hard_middle,
                bit_weight=bit_w, mae_weight=mae_w, mred_weight=mred_w,
                er_temperature=er_tau, cfg=loss_cfg, gemm_weight=gemm_w)
            entropy = model.bin_reg()
            collapse = collapse_regularizer(model, c_init)
            total = total + entropy_w * entropy + collapse_w * collapse
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            improved = False
            if (local % args.eval_every == 0) or local == epochs - 1:
                inits, metrics = evaluate_now()
                if metrics.objective_score < best['metrics'].objective_score - 1e-15:
                    save_best(stage, global_epoch, inits, metrics)
                    improved = True
                usage = quad_usage_stats(model)
                usage_track.append({'epoch': global_epoch, 'stage': stage, **usage})
                history.write(json.dumps({
                    'epoch': global_epoch, 'stage': stage, 'local': local,
                    'loss': float(total.detach().cpu()),
                    'entropy': float(entropy.detach().cpu()),
                    'collapse': float(collapse.detach().cpu()),
                    'intermediate_fraction': usage['intermediate_fraction'],
                    'hard_score': metrics.objective_score, 'hard_wce': metrics.WCE,
                    'hard_wnmse': metrics.workload_NMSE,
                    'terms': {k: float(v.detach().cpu()) for k, v in terms.items()},
                }) + '\n')
                history.flush()
                if (local % args.print_every == 0) or local == epochs - 1 or improved:
                    print(f'[{stage}:{local:05d}] loss={float(total):.6f} tau={args.tau0 * c_init:.2f} '
                          f'collapse_w={collapse_w:.3f} mid_frac={usage["intermediate_fraction"]:.3f} '
                          f'hard={metrics.short()}{" *BEST*" if improved else ""}', flush=True)
            global_epoch += 1

    start = time.time()
    run_phase('quad_warmup_bit', args.warmup_epochs, hard_middle=False, lr=args.lr, warmup=True)
    run_phase('quad_soft', args.soft_epochs, hard_middle=False, lr=args.lr)
    run_phase('quad_hard_ste', args.hard_epochs, hard_middle=True, lr=args.lr * args.hard_lr_scale)

    pop_lr = args.lr if args.population_lr is None else args.population_lr
    for member in range(args.population_members):
        if member % 2 == 0:
            conf, noise = args.population_init_conf, args.population_noise_std
        else:
            conf, noise = args.population_strong_conf, args.population_strong_noise_std
        reset_quad_from_inits(model, design, best['inits'], conf, noise)
        run_phase(f'pop{member:02d}_soft', args.population_soft_epochs,
                  hard_middle=False, lr=pop_lr)
        run_phase(f'pop{member:02d}_hard', args.population_hard_epochs,
                  hard_middle=True, lr=pop_lr * args.hard_lr_scale)
        print(f'[pop{member:02d}] done (conf={conf}, noise={noise}); '
              f'global best {best["metrics"].short()}', flush=True)
    history.close()

    write_json(out / 'quad_stats.json', {'usage_track': usage_track, 'final': quad_usage_stats(model)})
    design.export_rtl(_UNIFIED / 'rtl_sources', out / 'best_rtl', best['inits'],
                      metadata={'metrics': best['metrics'].to_dict(), **artifact_common,
                                'stage': best['stage']})
    write_json(out / 'summary.json', {
        'design': design.spec.name, 'seed': args.seed,
        'parameterization': 'quad4_softmax',
        'best_stage': best['stage'], 'best_epoch': best['epoch'],
        'best_metrics': best['metrics'].to_dict(),
        'initial_metrics': initial_metrics.to_dict(),
        'final_quad_usage': quad_usage_stats(model),
        'elapsed_sec': time.time() - start,
        'train_args': vars(args),
    })
    print(f'[best] stage={best["stage"]} {best["metrics"].short()}')
    print(f'[best-json] {out / "best_signed88_inits.json"}')
    print(f'[best-rtl] {out / "best_rtl"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

from signed88.common import (
    GEMM_OBJECTIVE_SCHEMA, ObjectiveWeights, hamming, hex_to_int, read_json, set_seed,
    validate_objective_weights, write_json,
)
from signed88.data import load_calibration_csv, to_torch
from signed88.hardware import choices, get_design
from signed88.hard_search import gradient_ranked_hard_search, rank_model_gradients
from signed88.losses import LossConfig, compute_loss
from signed88.metrics import evaluate_design

ROOT = Path(__file__).resolve().parent


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,data):
        for s in self.streams: s.write(data); s.flush()
        return len(data)
    def flush(self):
        for s in self.streams: s.flush()


def parse_args():
    p=argparse.ArgumentParser(description='Unified distribution-aware signed8x8 LUT trainer')
    p.add_argument('--design',default='fast',choices=choices())
    p.add_argument('--calibration-csv',default=str(ROOT/'data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv'))
    p.add_argument('--calibration-weight-column',default='auto',choices=['auto','count','p_calib','weight','probability'])
    p.add_argument('--out-dir',default='runs/run_00')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='auto',help='auto | cpu | cuda | cuda:0 ...')
    p.add_argument('--rtl-template-root',default=str(ROOT/'rtl_sources'))

    p.add_argument(
        '--init-mode', default='random',
        choices=['random','random_logits','baseline','json','json_perturb'],
        help=(
            'random samples hard INIT bits then assigns one shared confidence; '
            'random_logits draws every mutable logit directly from a normal distribution'
        ),
    )
    p.add_argument('--base-inits-json')
    p.add_argument('--random-p',type=float,default=0.5)
    p.add_argument('--json-perturb-p',type=float,default=0.02)
    p.add_argument('--init-conf',type=float,default=0.55)
    p.add_argument('--init-noise-std',type=float,default=0.0)
    p.add_argument('--random-logit-mean',type=float,default=0.0,
                   help='mean of direct normal logit initialization in random_logits mode')
    p.add_argument('--random-logit-std',type=float,default=1.0,
                   help='standard deviation of direct normal logit initialization in random_logits mode')

    p.add_argument('--stage1-epochs',type=int,default=6000)
    p.add_argument('--stage2-epochs',type=int,default=10000)
    p.add_argument('--stage3-epochs',type=int,default=500)
    p.add_argument('--lr',type=float,default=0.002)
    p.add_argument('--stage3-lr-scale',type=float,default=0.03,
                   help='legacy scale used only by the population hard phase')
    p.add_argument('--stage3-lr',type=float,default=0.0002,
                   help='learning rate for each restarted hard-STE block')
    p.add_argument('--stage3-restart-conf',type=float,default=0.51,
                   help='map the current hard best to +/-logit(conf) before each hard block')
    p.add_argument('--stage3-block-epochs',type=int,default=25)
    p.add_argument('--stage3-no-progress-rounds',type=int,default=1)
    p.add_argument('--stage3-single-top-k',type=int,default=64,
                   help='gradient-ranked exact single flips per block; 0 evaluates all searchable bits')
    p.add_argument('--stage3-pair-top-k',type=int,default=56,
                   help='top-ranked bits used to form exact pairs; 0 disables pair search')
    p.add_argument('--stage3-pair-max-pairs',type=int,default=1540,
                   help='maximum exact pair candidates per block; 0 means no cap')
    p.add_argument(
        '--disable-stage3-exact-search','--disable-stage3-exact-single',
        dest='disable_stage3_exact_search',action='store_true',
        help='disable exact hard single/pair acceptance (the old single-only flag is an alias)',
    )
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--soft-c-init',type=float,default=1.0)
    p.add_argument('--soft-c-out',type=float,default=1.0)
    p.add_argument('--hard-c-init',type=float,default=1.0)
    p.add_argument('--hard-c-out',type=float,default=1.0)

    p.add_argument('--stage1-bit-weight',type=float,default=1.0)
    p.add_argument('--stage1-mae-weight',type=float,default=0.25)
    p.add_argument('--stage1-mred-weight',type=float,default=0.0)
    p.add_argument('--stage1-gemm-weight',type=float,default=0.02)
    p.add_argument('--stage2-bit-start',type=float,default=1.0)
    p.add_argument('--stage2-bit-end',type=float,default=0.05)
    p.add_argument('--stage2-mae-start',type=float,default=0.20)
    p.add_argument('--stage2-mae-end',type=float,default=0.02)
    p.add_argument('--stage2-mred-start',type=float,default=0.0001)
    p.add_argument('--stage2-mred-end',type=float,default=0.0001)
    p.add_argument('--stage2-gemm-start',type=float,default=0.02)
    p.add_argument('--stage2-gemm-end',type=float,default=1.0)
    p.add_argument('--stage3-bit-weight',type=float,default=0.005)
    p.add_argument('--stage3-mae-weight',type=float,default=0.04)
    p.add_argument('--stage3-mred-weight',type=float,default=0.0001)
    p.add_argument('--stage3-gemm-weight',type=float,default=1.0)

    p.add_argument('--calibration-mix',type=float,default=0.98)
    p.add_argument('--er-weight',type=float,default=0.0001)
    p.add_argument('--er-temperature-start',type=float,default=4.0)
    p.add_argument('--er-temperature-end',type=float,default=0.10)
    p.add_argument('--mse-weight',type=float,default=1.0,
                   help='weight of workload normalized MSE')
    p.add_argument('--bias-weight',type=float,default=1.0,
                   help='weight of coherent global squared-bias accumulation')
    p.add_argument('--conditional-bias-weight',type=float,default=0.001,
                   help='weight of signed-a/b conditional bias excess')
    p.add_argument('--bias-effective-k',type=float,default=1024.0,
                   help='representative GEMM inner dimension used to amplify bias^2')
    p.add_argument('--zero-weight',type=float,default=0.25)
    p.add_argument('--symmetry-weight',type=float,default=0.0)
    p.add_argument('--bin-weight',type=float,default=0.0)
    p.add_argument('--bit-weighting',default='linear',choices=['uniform','linear','sqrt_value','value'])
    p.add_argument('--wce-weight',type=float,default=0.0,
                   help='soft logsumexp worst-case-error surrogate weight over the 4096 LL states')
    p.add_argument('--wce-beta',type=float,default=0.25,
                   help='sharpness of the soft worst-case surrogate; larger tracks the true max more tightly')

    p.add_argument('--score-mse-weight',type=float,default=1.0)
    p.add_argument('--score-mred-weight',type=float,default=0.0001)
    p.add_argument('--score-er-weight',type=float,default=0.00005)
    p.add_argument('--score-ned-weight',type=float,default=0.05)
    p.add_argument('--score-bias-weight',type=float,default=1.0)
    p.add_argument('--score-conditional-bias-weight',type=float,default=0.001)
    p.add_argument('--score-uniform-mred-weight',type=float,default=0.0001)
    p.add_argument('--score-wce-weight',type=float,default=0.0,
                   help='hard objective weight of normalized uniform WCE (WCE/16384); WCE tier tracks PPL closely')

    p.add_argument('--population-size',type=int,default=24)
    p.add_argument('--population-flip-p',type=float,default=0.0007)
    p.add_argument('--population-epochs',type=int,default=700)
    p.add_argument('--population-soft-epochs',type=int,default=150)
    p.add_argument('--population-lr',type=float,default=0.00025)
    p.add_argument('--population-init-conf',type=float,default=0.53)
    p.add_argument('--population-noise-std',type=float,default=0.001)

    p.add_argument('--eval-every',type=int,default=25)
    p.add_argument('--print-every',type=int,default=25)
    return p.parse_args()


def resolve_device(text: str) -> torch.device:
    if text=='auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(text)


def read_inits(path: Path, design):
    obj=read_json(path)
    declared=obj.get('design') or obj.get('design_spec',{}).get('design')
    if declared and get_design(str(declared)).spec.name != design.spec.name:
        raise ValueError(f'JSON design {declared} != requested {design.spec.name}')
    return design.normalize_inits(obj.get('inits',obj))


def initial_inits(args, design):
    rng=random.Random(args.seed)
    if args.init_mode=='baseline': return design.normalize_inits(design.spec.base_inits)
    if args.init_mode=='random': return design.random_inits(args.random_p,rng)
    if args.init_mode=='random_logits': return design.normalize_inits(design.spec.base_inits)
    if not args.base_inits_json: raise ValueError(f'--init-mode {args.init_mode} requires --base-inits-json')
    base=read_inits(Path(args.base_inits_json),design)
    if args.init_mode=='json': return base
    return design.perturb_inits(base,args.json_perturb_p,rng,force_change=True)


def lerp(a,b,t): return float(a)+(float(b)-float(a))*float(t)


def exact_single_step(design, base_inits, rankings, profile, objective, top_k):
    """Exactly evaluate gradient-ranked hard single flips around ``base_inits``.

    Gradients only choose the evaluation order/subset.  Acceptance is based
    solely on the discrete NumPy hardware model and the configured objective.
    ``top_k=0`` means the complete searchable single-bit neighbourhood.
    """
    limit = len(rankings) if int(top_k) == 0 else min(int(top_k), len(rankings))
    if limit < 0:
        raise ValueError('top_k must be non-negative')
    return gradient_ranked_hard_search(
        design,
        base_inits,
        rankings,
        evaluate=lambda inits: evaluate_design(design,inits,profile,objective),
        better=lambda new,old: new.objective_score < old.objective_score-1e-15,
        top_k=limit,
        pair_top_k=0,
        max_pairs=0,
    )


def exact_hard_step(
    design,base_inits,rankings,profile,objective,
    single_top_k,pair_top_k,pair_max_pairs,
):
    """Exactly select the best improving hard single or pair flip.

    The differentiable gradient only orders or truncates the candidate set.
    Every candidate is evaluated with the discrete NumPy hardware model and
    only a strict decrease of the configured deployment objective is accepted.
    ``single_top_k=0`` scans every searchable single bit, ``pair_top_k=0``
    disables pairs, and ``pair_max_pairs=0`` removes the pair-count cap.
    """
    single_limit=(
        len(rankings) if int(single_top_k)==0
        else min(int(single_top_k),len(rankings))
    )
    pair_limit=min(int(pair_top_k),len(rankings))
    if single_limit<0 or pair_limit<0 or int(pair_max_pairs)<0:
        raise ValueError('Stage3 exact-search limits must be non-negative')
    max_pairs=None if int(pair_max_pairs)==0 else int(pair_max_pairs)
    return gradient_ranked_hard_search(
        design,
        base_inits,
        rankings,
        evaluate=lambda inits:evaluate_design(design,inits,profile,objective),
        better=lambda new,old:new.objective_score < old.objective_score-1e-15,
        top_k=single_limit,
        pair_top_k=pair_limit,
        max_pairs=max_pairs,
    )


def main():
    args=parse_args(); design=get_design(args.design); args.design=design.spec.name
    out=Path(args.out_dir).resolve()
    protected=('summary.json','best_signed88_inits.json','history.jsonl','population_summary.json','best_rtl')
    existing=[name for name in protected if (out/name).exists()]
    if existing:
        raise FileExistsError(
            f'output directory already contains protected artifact(s): {existing}; '
            'use a new --out-dir'
        )
    out.mkdir(parents=True,exist_ok=True)
    log_f=(out/'terminal_log.txt').open('w',encoding='utf-8'); old_out,old_err=sys.stdout,sys.stderr; sys.stdout=Tee(old_out,log_f);sys.stderr=sys.stdout
    try:
        set_seed(args.seed); device=resolve_device(args.device)
        profile=load_calibration_csv(Path(args.calibration_csv),args.calibration_weight_column)
        batch=to_torch(profile,device)
        if not math.isfinite(args.calibration_mix) or not 0.0 <= args.calibration_mix <= 1.0:
            raise ValueError('--calibration-mix must be in [0,1]')
        if not math.isfinite(args.bias_effective_k) or args.bias_effective_k < 1.0:
            raise ValueError('--bias-effective-k must be >= 1')
        if not math.isfinite(args.random_logit_mean):
            raise ValueError('--random-logit-mean must be finite')
        if not math.isfinite(args.random_logit_std) or args.random_logit_std <= 0.0:
            raise ValueError('--random-logit-std must be finite and > 0')
        if args.init_mode == 'random_logits' and args.init_noise_std != 0.0:
            raise ValueError('--init-noise-std must be 0 in random_logits mode')
        if not math.isfinite(args.stage3_restart_conf) or not 0.5 < args.stage3_restart_conf < 1.0:
            raise ValueError('--stage3-restart-conf must be strictly between 0.5 and 1')
        if not math.isfinite(args.stage3_lr) or args.stage3_lr <= 0.0:
            raise ValueError('--stage3-lr must be finite and > 0')
        if args.stage3_block_epochs <= 0:
            raise ValueError('--stage3-block-epochs must be > 0')
        if args.stage3_no_progress_rounds <= 0:
            raise ValueError('--stage3-no-progress-rounds must be > 0')
        if args.stage3_single_top_k < 0:
            raise ValueError('--stage3-single-top-k must be >= 0')
        if args.stage3_pair_top_k < 0:
            raise ValueError('--stage3-pair-top-k must be >= 0')
        if args.stage3_pair_max_pairs < 0:
            raise ValueError('--stage3-pair-max-pairs must be >= 0')
        for name in (
            'mse_weight','bias_weight','conditional_bias_weight','er_weight',
            'zero_weight','symmetry_weight','bin_weight','stage1_bit_weight',
            'stage1_mae_weight','stage1_mred_weight','stage2_bit_start',
            'stage2_bit_end','stage2_mae_start','stage2_mae_end',
            'stage2_mred_start','stage2_mred_end','stage3_bit_weight',
            'stage3_mae_weight','stage3_mred_weight','stage1_gemm_weight',
            'stage2_gemm_start','stage2_gemm_end','stage3_gemm_weight',
            'wce_weight','score_wce_weight',
        ):
            if not math.isfinite(float(getattr(args,name))) or float(getattr(args,name)) < 0.0:
                raise ValueError(f'--{name.replace("_","-")} must be non-negative')
        if not math.isfinite(args.wce_beta) or args.wce_beta <= 0.0:
            raise ValueError('--wce-beta must be finite and > 0')
        objective=ObjectiveWeights(
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
        loss_cfg=LossConfig(
            calibration_mix=args.calibration_mix,
            er_weight=args.er_weight,
            mse_weight=args.mse_weight,
            bias_weight=args.bias_weight,
            conditional_bias_weight=args.conditional_bias_weight,
            bias_effective_k=args.bias_effective_k,
            zero_weight=args.zero_weight,
            symmetry_weight=args.symmetry_weight,
            bin_weight=args.bin_weight,
            bit_weighting=args.bit_weighting,
            wce_weight=args.wce_weight,
            wce_beta=args.wce_beta,
        )
        requested_base=initial_inits(args,design)
        model=design.build_model(requested_base,args.init_conf,args.init_noise_std)
        initialization={'mode':args.init_mode}
        if args.init_mode=='random_logits':
            logit_generator=torch.Generator(device='cpu')
            logit_generator.manual_seed(args.seed)
            initialization.update(model.randomize_mutable_logits(
                mean=args.random_logit_mean,
                std=args.random_logit_std,
                generator=logit_generator,
            ))
            initialization['rng_seed']=args.seed
            expected=sum(len(design.spec.mutable_bits[name]) for name in design.spec.train_names)
            if initialization['mutable_logit_count'] != expected:
                raise AssertionError(
                    f'randomized {initialization["mutable_logit_count"]} mutable logits, expected {expected}'
                )
        # The actual model thresholds are the only truthful initial hard INITs.
        # This also keeps nonzero --init-noise-std runs internally consistent.
        base=design.normalize_inits(model.hard_inits())
        initialization['initial_hard_hamming_from_requested'] = hamming(
            requested_base, base, design.spec.mutable_bits,
        )
        initialization['initial_hard_one_count'] = sum(
            (hex_to_int(base[name]) >> bit) & 1
            for name in design.spec.train_names
            for bit in design.spec.mutable_bits[name]
        )
        base_metrics=evaluate_design(design,base,profile,objective)
        print(f'[design] {design.spec.name} resources={design.spec.resource_summary}')
        print(f'[device] {device}')
        print(f'[calibration] rows={profile.row_count} coverage={int((profile.state_probability>0).sum())}/4096 zeroP={profile.zero_probability:.6f}')
        print(f'[semantics] direct signed-int8 rows -> final signed-int16 loss; 4096 LL states are only an internal cache')
        print(f'[gemm-loss] K_eff={args.bias_effective_k:g} mse={args.mse_weight:g} global_bias={args.bias_weight:g} conditional_bias={args.conditional_bias_weight:g}')
        if args.wce_weight>0 or args.score_wce_weight>0:
            print(f'[wce] soft_weight={args.wce_weight:g} beta={args.wce_beta:g} score_weight={args.score_wce_weight:g}')
        if args.init_mode=='random_logits':
            print(
                '[initialization] random_logits '
                f'N({args.random_logit_mean:g},{args.random_logit_std:g}^2) '
                f'mutable={initialization["mutable_logit_count"]} '
                f'tables={initialization["trainable_lut_tables"]} '
                f'sample_mean={initialization["sample_mean"]:.5f} '
                f'sample_std={initialization["sample_std"]:.5f} '
                f'positive={initialization["positive_fraction"]:.3f}'
            )
        print(f'[initial] {base_metrics.short()}')

        model=model.to(device)
        opt=torch.optim.Adam(model.parameters(),lr=args.lr)
        history_path=out/'history.jsonl'
        best={'inits':base,'metrics':base_metrics.to_dict(),'stage':'initial','epoch':-1}
        total_main=max(1,args.stage1_epochs+args.stage2_epochs+args.stage3_epochs); global_epoch=0

        def save_best(inits,metrics,stage,local_epoch,loss_value=None,terms=None):
            nonlocal best
            if metrics.objective_score >= float(best['metrics']['objective_score'])-1e-15: return False
            best={'inits':inits,'metrics':metrics.to_dict(),'stage':stage,'epoch':global_epoch,'local_epoch':local_epoch,'loss':loss_value,'terms':terms}
            artifact=design.artifact(inits,metrics=metrics.to_dict(),extra={
                'stage':stage,'global_epoch':global_epoch,'local_epoch':local_epoch,'seed':args.seed,
                'calibration':profile.metadata(),'objective_schema':GEMM_OBJECTIVE_SCHEMA,
                'objective_weights':objective.__dict__,'train_args':vars(args),
                'initialization':initialization,
            })
            write_json(out/'best_signed88_inits.json',artifact)
            design.export_rtl(Path(args.rtl_template_root),out/'best_rtl',inits,metadata={
                'metrics':metrics.to_dict(),'calibration':profile.metadata(),
                'objective_schema':GEMM_OBJECTIVE_SCHEMA,
                'objective_weights':objective.__dict__,'initialization':initialization,
            })
            return True

        # Always materialize the initial artifact so zero-epoch smoke runs are useful.
        write_json(out/'initial_signed88_inits.json',design.artifact(base,metrics=base_metrics.to_dict(),extra={
            'seed':args.seed,'calibration':profile.metadata(),'objective_schema':GEMM_OBJECTIVE_SCHEMA,
            'objective_weights':objective.__dict__,'initialization':initialization,
            'train_args':vars(args),
        }))
        if not (out/'best_signed88_inits.json').exists():
            write_json(out/'best_signed88_inits.json',design.artifact(base,metrics=base_metrics.to_dict(),extra={
                'stage':'initial','seed':args.seed,'calibration':profile.metadata(),
                'objective_schema':GEMM_OBJECTIVE_SCHEMA,
                'objective_weights':objective.__dict__,'train_args':vars(args),
                'initialization':initialization,
            }))
            design.export_rtl(Path(args.rtl_template_root),out/'best_rtl',base,metadata={
                'metrics':base_metrics.to_dict(),'calibration':profile.metadata(),
                'objective_schema':GEMM_OBJECTIVE_SCHEMA,
                'objective_weights':objective.__dict__,'initialization':initialization,
            })

        def train_phase(stage,epochs,*,hard_middle,lr,bit0,bit1,mae0,mae1,mred0,mred1,gemm0,gemm1,c_init,c_out,model_ref,opt_ref,extra=None):
            nonlocal global_epoch
            if epochs<=0: return
            for g in opt_ref.param_groups: g['lr']=lr
            for local in range(epochs):
                t=0.0 if epochs<=1 else local/(epochs-1)
                bit_w,mae_w,mred_w=lerp(bit0,bit1,t),lerp(mae0,mae1,t),lerp(mred0,mred1,t)
                gemm_w=lerp(gemm0,gemm1,t)
                main_t=min(1.0,global_epoch/max(total_main-1,1)); tau=lerp(args.er_temperature_start,args.er_temperature_end,main_t)
                opt_ref.zero_grad(set_to_none=True)
                loss,terms=compute_loss(model_ref,batch,c_init=c_init,c_out=c_out,hard_middle=hard_middle,bit_weight=bit_w,mae_weight=mae_w,mred_weight=mred_w,er_temperature=tau,cfg=loss_cfg,gemm_weight=gemm_w)
                loss.backward()
                if args.grad_clip>0: torch.nn.utils.clip_grad_norm_(model_ref.parameters(),args.grad_clip)
                opt_ref.step()
                should_eval=(local==epochs-1 or local%max(1,args.eval_every)==0 or global_epoch%max(1,args.eval_every)==0)
                if should_eval:
                    hard=model_ref.hard_inits(); metrics=evaluate_design(design,hard,profile,objective); terms_f={k:float(v.detach().cpu()) for k,v in terms.items()}; loss_f=float(loss.detach().cpu())
                    improved=save_best(hard,metrics,stage,local,loss_f,terms_f)
                    row={'global_epoch':global_epoch,'stage':stage,'local_epoch':local,'loss':loss_f,'terms':terms_f,'metrics':metrics.to_dict(),'improved':improved,'weights':{'bit':bit_w,'mae':mae_w,'mred':mred_w,'gemm':gemm_w,'er_tau':tau},'extra':extra or {}}
                    with history_path.open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
                    if improved or local%max(1,args.print_every)==0 or local==epochs-1:
                        print(f'[epoch {global_epoch:06d} {stage}:{local:05d}] loss={loss_f:.7f} bit={terms_f["bit"]:.5f} gemm_w={gemm_w:.4g} nmse={terms_f["nmse"]:.6g} gbias={terms_f["bias"]:.6g} cbias={terms_f["conditional_bias"]:.6g} mred={terms_f["mred"]:.5f} er={terms_f["er"]:.5f} mae={terms_f["mae"]:.5f} hard_{metrics.short()}{" *BEST*" if improved else ""}')
                global_epoch+=1

        train_phase('stage1_soft_bit',args.stage1_epochs,hard_middle=False,lr=args.lr,bit0=args.stage1_bit_weight,bit1=args.stage1_bit_weight,mae0=args.stage1_mae_weight,mae1=args.stage1_mae_weight,mred0=args.stage1_mred_weight,mred1=args.stage1_mred_weight,gemm0=args.stage1_gemm_weight,gemm1=args.stage1_gemm_weight,c_init=args.soft_c_init,c_out=args.soft_c_out,model_ref=model,opt_ref=opt)
        train_phase('stage2_soft_signed_ramp',args.stage2_epochs,hard_middle=False,lr=args.lr,bit0=args.stage2_bit_start,bit1=args.stage2_bit_end,mae0=args.stage2_mae_start,mae1=args.stage2_mae_end,mred0=args.stage2_mred_start,mred1=args.stage2_mred_end,gemm0=args.stage2_gemm_start,gemm1=args.stage2_gemm_end,c_init=args.soft_c_init,c_out=args.soft_c_out,model_ref=model,opt_ref=opt)
        searchable_bit_count=sum(
            len(design.spec.search_bits[name]) for name in design.spec.train_names
        )
        requested_single_count=(
            searchable_bit_count if args.stage3_single_top_k==0
            else min(args.stage3_single_top_k,searchable_bit_count)
        )
        requested_pair_bits=min(args.stage3_pair_top_k,searchable_bit_count)
        requested_pair_count=requested_pair_bits*(requested_pair_bits-1)//2
        if args.stage3_pair_max_pairs>0:
            requested_pair_count=min(requested_pair_count,args.stage3_pair_max_pairs)
        stage3_info={
            'requested_epochs':args.stage3_epochs,'completed_epochs':0,'rounds':0,
            'restart_conf':args.stage3_restart_conf,'learning_rate':args.stage3_lr,
            'model_restarts':0,'searchable_bits':searchable_bit_count,
            'full_single_neighborhood':requested_single_count==searchable_bit_count,
            'full_pair_neighborhood':(
                requested_pair_bits==searchable_bit_count
                and requested_pair_count==searchable_bit_count*(searchable_bit_count-1)//2
            ),
            'exact_single_evaluations':0,'exact_single_accepts':0,
            'exact_hard_evaluations':0,'exact_single_candidates':0,
            'exact_pair_candidates':0,'exact_pair_accepts':0,
            'hard_training_accepts':0,'termination':'disabled' if args.stage3_epochs<=0 else 'budget_exhausted',
        }
        remaining=max(0,args.stage3_epochs); no_progress=0; stage3_round=0
        hard_model=None; hard_opt=None; hard_anchor_inits=None
        while remaining>0:
            block=min(args.stage3_block_epochs,remaining)
            round_start_score=float(best['metrics']['objective_score'])
            continued_model=hard_model is not None
            if hard_model is None:
                hard_anchor_inits=design.normalize_inits(best['inits'])
                hard_model=design.build_model(
                    hard_anchor_inits,args.stage3_restart_conf,0.0,
                )
                if design.normalize_inits(hard_model.hard_inits()) != hard_anchor_inits:
                    raise AssertionError('Stage3 restart changed the hard best INIT')
                hard_model=hard_model.to(device)
                hard_opt=torch.optim.Adam(hard_model.parameters(),lr=args.stage3_lr)
                stage3_info['model_restarts']+=1
            elif design.normalize_inits(hard_model.hard_inits()) != hard_anchor_inits:
                raise AssertionError('continued Stage3 model no longer matches its hard anchor')
            stage_name=f'stage3_hard_ste_r{stage3_round:02d}'
            train_phase(
                stage_name,block,hard_middle=True,lr=args.stage3_lr,
                bit0=args.stage3_bit_weight,bit1=args.stage3_bit_weight,
                mae0=args.stage3_mae_weight,mae1=args.stage3_mae_weight,
                mred0=args.stage3_mred_weight,mred1=args.stage3_mred_weight,
                gemm0=args.stage3_gemm_weight,gemm1=args.stage3_gemm_weight,
                c_init=args.hard_c_init,c_out=args.hard_c_out,
                model_ref=hard_model,opt_ref=hard_opt,
                extra={
                    'stage3_round':stage3_round,
                    'restart_conf':args.stage3_restart_conf,
                    'continued_model':continued_model,
                },
            )
            remaining-=block; stage3_info['completed_epochs']+=block; stage3_info['rounds']+=1
            hard_improved=float(best['metrics']['objective_score']) < round_start_score-1e-15
            if hard_improved: stage3_info['hard_training_accepts']+=1
            live_hard_inits=design.normalize_inits(hard_model.hard_inits())
            live_hard_drifted=live_hard_inits!=hard_anchor_inits

            exact_accepted=False
            if not args.disable_stage3_exact_search:
                search_base=design.normalize_inits(best['inits'])
                rank_model=design.build_model(search_base,args.stage3_restart_conf,0.0).to(device)
                rank_model.zero_grad(set_to_none=True)
                rank_loss,_=compute_loss(
                    rank_model,batch,c_init=args.hard_c_init,c_out=args.hard_c_out,
                    hard_middle=True,bit_weight=args.stage3_bit_weight,
                    mae_weight=args.stage3_mae_weight,mred_weight=args.stage3_mred_weight,
                    er_temperature=args.er_temperature_end,cfg=loss_cfg,
                    gemm_weight=args.stage3_gemm_weight,
                )
                rank_loss.backward()
                rankings=rank_model_gradients(
                    rank_model,design,search_base,normalization='table',
                    c_init=args.hard_c_init,
                )
                search=exact_hard_step(
                    design,search_base,rankings,profile,objective,
                    args.stage3_single_top_k,args.stage3_pair_top_k,
                    args.stage3_pair_max_pairs,
                )
                single_candidates=sum(len(candidate.flips)==1 for candidate in search.candidates)
                pair_candidates=sum(len(candidate.flips)==2 for candidate in search.candidates)
                stage3_info['exact_hard_evaluations']+=search.evaluations
                stage3_info['exact_single_evaluations']+=single_candidates+1
                stage3_info['exact_single_candidates']+=single_candidates
                stage3_info['exact_pair_candidates']+=pair_candidates
                accepted=search.accepted
                if accepted is not None:
                    flips=[f'{name}[{bit}]' for name,bit in accepted.flips]
                    move_kind='single' if len(accepted.flips)==1 else 'pair'
                    exact_accepted=save_best(
                        accepted.inits,accepted.evaluation,
                        f'stage3_exact_{move_kind}_r{stage3_round:02d}',stage3_round,
                        None,{'flips':flips,'evaluations':search.evaluations},
                    )
                    if not exact_accepted:
                        raise AssertionError('exact Stage3 search accepted a non-improving candidate')
                    stage3_info[f'exact_{move_kind}_accepts']+=1
                    print(
                        f'[stage3 exact {move_kind} r{stage3_round:02d}] ACCEPT {"+".join(flips)} '
                        f'score={accepted.evaluation.objective_score:.10f} '
                        f'wMSE={accepted.evaluation.workload_MSE:.4f} '
                        f'wBias={accepted.evaluation.workload_bias:+.5f}'
                    )
                event_metrics=(accepted.evaluation if accepted is not None else search.base_evaluation)
                event_kind=(
                    'single' if accepted is not None and len(accepted.flips)==1
                    else 'pair' if accepted is not None
                    else 'search'
                )
                event={
                    'global_epoch':global_epoch,'event':'exact_hard_search',
                    'stage':f'stage3_exact_{event_kind}_r{stage3_round:02d}',
                    'local_epoch':stage3_round,'loss':None,'terms':{},
                    'metrics':event_metrics.to_dict(),'improved':exact_accepted,'weights':{},
                    'extra':{
                        'stage3_round':stage3_round,'evaluations':search.evaluations,
                        'single_candidates':single_candidates,
                        'pair_candidates':pair_candidates,
                        'accepted_flips':[] if accepted is None else [list(x) for x in accepted.flips],
                        'base_score':search.base_evaluation.objective_score,
                    },
                }
                with history_path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(event)+'\n')
                if accepted is None:
                    print(
                        f'[stage3 exact r{stage3_round:02d}] no improving hard move '
                        f'({single_candidates} singles, {pair_candidates} pairs)'
                    )

            if hard_improved or exact_accepted:
                no_progress=0
                # The accepted global best may be an intermediate Hard-STE
                # state or an exact discrete move.  Rebuild around that exact
                # state next round and discard stale Adam moments.
                hard_model=None; hard_opt=None; hard_anchor_inits=None
            else:
                no_progress+=1
                if no_progress>=args.stage3_no_progress_rounds:
                    stage3_info['termination']='no_hard_progress'
                    print(
                        f'[stage3] early stop after round {stage3_round}: '
                        f'no hard improvement for {no_progress} round(s)'
                    )
                    break
                if live_hard_drifted:
                    # Never accumulate from a rejected hard configuration.
                    # If no threshold was crossed, retain the same logits and
                    # optimizer so a multi-block patience setting is useful.
                    hard_model=None; hard_opt=None; hard_anchor_inits=None
            stage3_round+=1
        if (
            remaining<=0 and args.stage3_epochs>0
            and stage3_info['termination']!='no_hard_progress'
        ):
            stage3_info['termination']='budget_exhausted'

        population=[]
        if args.population_size>0 and args.population_epochs>0:
            for member in range(args.population_size):
                member_seed=args.seed+100000+member; rng=random.Random(member_seed); set_seed(member_seed)
                start=design.perturb_inits(best['inits'],args.population_flip_p,rng,force_change=True)
                pm=design.build_model(start,args.population_init_conf,args.population_noise_std).to(device); po=torch.optim.Adam(pm.parameters(),lr=args.population_lr)
                soft=min(args.population_soft_epochs,args.population_epochs); hard=max(0,args.population_epochs-soft)
                print(f'[population {member:02d}] seed={member_seed} hamming={hamming(best["inits"],start,design.spec.search_bits)}')
                train_phase(f'pop{member:02d}_soft',soft,hard_middle=False,lr=args.population_lr,bit0=args.stage2_bit_end,bit1=max(0,args.stage2_bit_end*.2),mae0=args.stage2_mae_end,mae1=0,mred0=args.stage2_mred_end,mred1=args.stage2_mred_end,gemm0=args.stage2_gemm_start,gemm1=args.stage2_gemm_end,c_init=args.soft_c_init,c_out=args.soft_c_out,model_ref=pm,opt_ref=po,extra={'population_member':member})
                train_phase(f'pop{member:02d}_hard',hard,hard_middle=True,lr=args.population_lr*args.stage3_lr_scale,bit0=args.stage3_bit_weight,bit1=args.stage3_bit_weight,mae0=args.stage3_mae_weight,mae1=args.stage3_mae_weight,mred0=args.stage3_mred_weight,mred1=args.stage3_mred_weight,gemm0=args.stage3_gemm_weight,gemm1=args.stage3_gemm_weight,c_init=args.hard_c_init,c_out=args.hard_c_out,model_ref=pm,opt_ref=po,extra={'population_member':member})
                final=pm.hard_inits(); fm=evaluate_design(design,final,profile,objective); population.append({'member':member,'seed':member_seed,'metrics':fm.to_dict(),'inits':final})
        write_json(out/'population_summary.json',{'population':population})
        best_metrics=evaluate_design(design,best['inits'],profile,objective)
        summary={'design':design.spec.name,'design_spec':design.spec.metadata(),'seed':args.seed,'device':str(device),'calibration':profile.metadata(),'objective_schema':GEMM_OBJECTIVE_SCHEMA,'objective_weights':objective.__dict__,'initialization':initialization,'initial_metrics':base_metrics.to_dict(),'best_metrics':best_metrics.to_dict(),'best_stage':best['stage'],'stage3':stage3_info,'best_json':str(out/'best_signed88_inits.json'),'best_rtl':str(out/'best_rtl'),'train_args':vars(args)}
        write_json(out/'summary.json',summary)
        print(f'[best] {best_metrics.short()}')
        print(f'[best-json] {out/"best_signed88_inits.json"}')
        print(f'[best-rtl] {out/"best_rtl"}')
        return 0
    finally:
        sys.stdout.flush(); sys.stdout=old_out; sys.stderr=old_err; log_f.close()

if __name__=='__main__': raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

from signed88.common import ObjectiveWeights, hex_to_int, int_to_hex, read_json, write_json
from signed88.data import load_calibration_csv
from signed88.hardware import choices, get_design
from signed88.metrics import evaluate_design

ROOT=Path(__file__).resolve().parent


def parse_args():
    p=argparse.ArgumentParser(description='Topology-independent hard INIT refinement')
    p.add_argument('--base-inits-json',required=True)
    p.add_argument('--design',default='auto',choices=('auto',)+choices())
    p.add_argument('--calibration-csv',default=None,help='default: inherit tagged training artifact, else packaged CSV')
    p.add_argument('--calibration-weight-column',default=None,choices=['auto','count','p_calib','weight','probability'])
    p.add_argument('--out-dir',default='refine_run')
    p.add_argument('--rtl-template-root',default=str(ROOT/'rtl_sources'))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--bit-rounds',type=int,default=20)
    p.add_argument('--pair-rounds',type=int,default=2)
    p.add_argument('--pair-candidate-bits',type=int,default=80)
    p.add_argument('--pair-max-pairs',type=int,default=3000)
    p.add_argument('--neutral-bits',type=int,default=32)
    p.add_argument('--basin-iters',type=int,default=10)
    p.add_argument('--basin-flip-min',type=int,default=2)
    p.add_argument('--basin-flip-max',type=int,default=4)
    p.add_argument('--basin-candidate-bits',type=int,default=96)
    p.add_argument('--basin-polish-rounds',type=int,default=2)
    p.add_argument('--basin-max-start-factor',type=float,default=1.20)
    p.add_argument('--max-wce',type=int,default=0)
    p.add_argument('--max-workload-mred',type=float,default=-1)
    p.add_argument('--max-workload-er',type=float,default=-1)
    p.add_argument('--score-mred-weight',type=float,default=None)
    p.add_argument('--score-er-weight',type=float,default=None)
    p.add_argument('--score-ned-weight',type=float,default=None)
    p.add_argument('--score-bias-weight',type=float,default=None)
    p.add_argument('--score-uniform-mred-weight',type=float,default=None)
    return p.parse_args()


def resolve_design(obj,requested):
    declared=obj.get('design') or obj.get('design_spec',{}).get('design')
    if requested=='auto':
        if not declared: raise ValueError('auto design requires tagged JSON')
        return get_design(declared)
    d=get_design(requested)
    if declared and get_design(declared).spec.name!=d.spec.name: raise ValueError('design mismatch')
    return d


def all_bits(design):
    return [(name,int(bit)) for name in design.spec.train_names for bit in design.spec.search_bits[name]]


def key(inits,design): return tuple(hex_to_int(inits[n]) for n in design.spec.train_names)


def flip(inits,flips):
    ints={k:hex_to_int(v) for k,v in inits.items()}
    for name,bit in flips: ints[name]^=1<<bit
    return {k:int_to_hex(v) for k,v in ints.items()}


def valid(m,args):
    if args.max_wce>0 and m.WCE>args.max_wce: return False
    if args.max_workload_mred>=0 and m.workload_MRED>args.max_workload_mred: return False
    if args.max_workload_er>=0 and m.workload_ER>args.max_workload_er: return False
    return True


def main():
    args=parse_args(); src=read_json(Path(args.base_inits_json)); design=resolve_design(src,args.design); args.design=design.spec.name
    inits=design.normalize_inits(src.get('inits',src))
    inherited_cal=src.get('calibration',{}) if isinstance(src.get('calibration',{}),dict) else {}
    cal_path=args.calibration_csv or inherited_cal.get('source') or str(ROOT/'data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv')
    if not Path(cal_path).exists(): cal_path=str(ROOT/'data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv')
    cal_col=args.calibration_weight_column or inherited_cal.get('weight_column') or 'auto'
    profile=load_calibration_csv(Path(cal_path),cal_col)
    inherited_obj=src.get('objective_weights',{}) if isinstance(src.get('objective_weights',{}),dict) else {}
    objective=ObjectiveWeights(
        inherited_obj.get('workload_mred',1.0) if args.score_mred_weight is None else args.score_mred_weight,
        inherited_obj.get('workload_er',0.25) if args.score_er_weight is None else args.score_er_weight,
        inherited_obj.get('workload_ned',0.10) if args.score_ned_weight is None else args.score_ned_weight,
        inherited_obj.get('workload_bias',0.05) if args.score_bias_weight is None else args.score_bias_weight,
        inherited_obj.get('uniform_mred',0.05) if args.score_uniform_mred_weight is None else args.score_uniform_mred_weight,
    )
    out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True); rng=random.Random(args.seed); cache={}
    def eval_cached(x):
        k=key(x,design)
        if k not in cache: cache[k]=evaluate_design(design,x,profile,objective)
        return cache[k]
    current=eval_cached(inits); steps=[]; bits=all_bits(design)
    print(f'[design] {design.spec.name} search_bits={len(bits)}')
    print(f'[start] {current.short()}')

    def record(stage,flips_,old,new):
        row={'stage':stage,'flips':[f'{n}:{b}' for n,b in flips_],'old_score':old.objective_score,'new_score':new.objective_score,'metrics':new.to_dict()};steps.append(row);write_json(out/'steps.json',{'steps':steps});print(f'[{stage}] ACCEPT {row["flips"]} {old.objective_score:.10f}->{new.objective_score:.10f} wMRED={new.workload_MRED:.10f}')

    def scan_single(base,base_m,scan_bits):
        scores=[]
        for bit in scan_bits:
            trial=flip(base,[bit]); m=eval_cached(trial); scores.append((m.objective_score-base_m.objective_score,bit,m))
        scores.sort(key=lambda x:(x[0],x[2].WCE))
        return scores

    for r in range(args.bit_rounds):
        best=None
        for delta,bit,m in scan_single(inits,current,bits):
            if delta < -1e-15 and valid(m,args): best=(bit,m);break
        if not best: print(f'[single] round={r+1} no improvement');break
        bit,m=best; old=current; inits=flip(inits,[bit]); current=m; record('single',[bit],old,current)

    for r in range(args.pair_rounds):
        scores=scan_single(inits,current,bits)
        ranked=[x[1] for x in scores]
        neutral=[x[1] for x in sorted(scores,key=lambda x:abs(x[0]))[:args.neutral_bits]]
        cand=[]
        for b in ranked+neutral:
            if b not in cand: cand.append(b)
            if len(cand)>=args.pair_candidate_bits: break
        pairs=list(itertools.combinations(cand,2))
        pairs.sort(key=lambda p: next(x[0] for x in scores if x[1]==p[0])+next(x[0] for x in scores if x[1]==p[1]))
        if len(pairs)>args.pair_max_pairs:
            head=pairs[:args.pair_max_pairs//2]; tail=pairs[args.pair_max_pairs//2:]; pairs=head+rng.sample(tail,min(len(tail),args.pair_max_pairs-len(head)))
        best=None
        for pair in pairs:
            trial=flip(inits,pair);m=eval_cached(trial)
            if valid(m,args) and m.objective_score < (best[1].objective_score if best else current.objective_score)-1e-15: best=(pair,m)
        if not best: print(f'[pair] round={r+1} no improvement');break
        pair,m=best;old=current;inits=flip(inits,pair);current=m;record('pair',pair,old,current)

    for it in range(args.basin_iters):
        scores=scan_single(inits,current,bits)
        pool=[]
        for _,bit,_ in scores:
            if bit not in pool: pool.append(bit)
            if len(pool)>=min(args.basin_candidate_bits,len(bits)): break
        if len(pool)<args.basin_flip_min: break
        k=rng.randint(args.basin_flip_min,min(args.basin_flip_max,len(pool))); perturb=rng.sample(pool,k); local=flip(inits,perturb); lm=eval_cached(local)
        if lm.objective_score > current.objective_score*args.basin_max_start_factor: continue
        for pr in range(args.basin_polish_rounds):
            best=None
            for delta,bit,m in scan_single(local,lm,pool):
                if delta < -1e-15 and valid(m,args): best=(bit,m);break
            if not best: break
            bit,m=best;local=flip(local,[bit]);lm=m
        if valid(lm,args) and lm.objective_score<current.objective_score-1e-15:
            old=current;inits=local;current=lm;record('basin',perturb,old,current)

    artifact=design.artifact(inits,metrics=current.to_dict(),extra={'stage':'refined','base_json':str(Path(args.base_inits_json).resolve()),'calibration':profile.metadata(),'objective_weights':objective.__dict__,'refine_args':vars(args),'accepted_steps':steps})
    write_json(out/'best_signed88_inits.json',artifact);design.export_rtl(Path(args.rtl_template_root),out/'best_rtl',inits,metadata={'metrics':current.to_dict(),'calibration':profile.metadata(),'objective_weights':objective.__dict__})
    print(f'[final] {current.short()}');print(f'[json] {out/"best_signed88_inits.json"}');return 0
if __name__=='__main__': raise SystemExit(main())

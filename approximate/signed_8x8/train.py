#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from signed88.common import ObjectiveWeights, hamming, read_json, set_seed, write_json
from signed88.data import load_calibration_csv, to_torch
from signed88.hardware import choices, get_design
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

    p.add_argument('--init-mode',default='random',choices=['random','baseline','json','json_perturb'])
    p.add_argument('--base-inits-json')
    p.add_argument('--random-p',type=float,default=0.5)
    p.add_argument('--json-perturb-p',type=float,default=0.02)
    p.add_argument('--init-conf',type=float,default=0.55)
    p.add_argument('--init-noise-std',type=float,default=0.0)

    p.add_argument('--stage1-epochs',type=int,default=6000)
    p.add_argument('--stage2-epochs',type=int,default=10000)
    p.add_argument('--stage3-epochs',type=int,default=500)
    p.add_argument('--lr',type=float,default=0.002)
    p.add_argument('--stage3-lr-scale',type=float,default=0.03)
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--soft-c-init',type=float,default=1.0)
    p.add_argument('--soft-c-out',type=float,default=1.0)
    p.add_argument('--hard-c-init',type=float,default=1.0)
    p.add_argument('--hard-c-out',type=float,default=1.0)

    p.add_argument('--stage1-bit-weight',type=float,default=1.0)
    p.add_argument('--stage1-mae-weight',type=float,default=0.25)
    p.add_argument('--stage1-mred-weight',type=float,default=0.0)
    p.add_argument('--stage2-bit-start',type=float,default=1.0)
    p.add_argument('--stage2-bit-end',type=float,default=0.05)
    p.add_argument('--stage2-mae-start',type=float,default=0.20)
    p.add_argument('--stage2-mae-end',type=float,default=0.02)
    p.add_argument('--stage2-mred-start',type=float,default=0.10)
    p.add_argument('--stage2-mred-end',type=float,default=1.0)
    p.add_argument('--stage3-bit-weight',type=float,default=0.08)
    p.add_argument('--stage3-mae-weight',type=float,default=0.04)
    p.add_argument('--stage3-mred-weight',type=float,default=1.0)

    p.add_argument('--calibration-mix',type=float,default=0.98)
    p.add_argument('--er-weight',type=float,default=0.25)
    p.add_argument('--er-temperature-start',type=float,default=4.0)
    p.add_argument('--er-temperature-end',type=float,default=0.10)
    p.add_argument('--bias-weight',type=float,default=0.05)
    p.add_argument('--zero-weight',type=float,default=0.25)
    p.add_argument('--symmetry-weight',type=float,default=0.0)
    p.add_argument('--bin-weight',type=float,default=0.0)
    p.add_argument('--bit-weighting',default='linear',choices=['uniform','linear','sqrt_value','value'])

    p.add_argument('--score-mred-weight',type=float,default=1.0)
    p.add_argument('--score-er-weight',type=float,default=0.25)
    p.add_argument('--score-ned-weight',type=float,default=0.10)
    p.add_argument('--score-bias-weight',type=float,default=0.05)
    p.add_argument('--score-uniform-mred-weight',type=float,default=0.05)

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
    if not args.base_inits_json: raise ValueError(f'--init-mode {args.init_mode} requires --base-inits-json')
    base=read_inits(Path(args.base_inits_json),design)
    if args.init_mode=='json': return base
    return design.perturb_inits(base,args.json_perturb_p,rng,force_change=True)


def lerp(a,b,t): return float(a)+(float(b)-float(a))*float(t)


def main():
    args=parse_args(); design=get_design(args.design); args.design=design.spec.name
    out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    log_f=(out/'terminal_log.txt').open('w',encoding='utf-8'); old_out,old_err=sys.stdout,sys.stderr; sys.stdout=Tee(old_out,log_f);sys.stderr=sys.stdout
    try:
        set_seed(args.seed); device=resolve_device(args.device)
        profile=load_calibration_csv(Path(args.calibration_csv),args.calibration_weight_column)
        batch=to_torch(profile,device)
        objective=ObjectiveWeights(args.score_mred_weight,args.score_er_weight,args.score_ned_weight,args.score_bias_weight,args.score_uniform_mred_weight)
        loss_cfg=LossConfig(args.calibration_mix,args.er_weight,args.bias_weight,args.zero_weight,args.symmetry_weight,args.bin_weight,args.bit_weighting)
        base=initial_inits(args,design); base_metrics=evaluate_design(design,base,profile,objective)
        print(f'[design] {design.spec.name} resources={design.spec.resource_summary}')
        print(f'[device] {device}')
        print(f'[calibration] rows={profile.row_count} coverage={int((profile.state_probability>0).sum())}/4096 zeroP={profile.zero_probability:.6f}')
        print(f'[semantics] direct signed-int8 rows -> final signed-int16 loss; 4096 LL states are only an internal cache')
        print(f'[initial] {base_metrics.short()}')

        model=design.build_model(base,args.init_conf,args.init_noise_std).to(device)
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
                'calibration':profile.metadata(),'objective_weights':objective.__dict__,'train_args':vars(args),
            })
            write_json(out/'best_signed88_inits.json',artifact)
            design.export_rtl(Path(args.rtl_template_root),out/'best_rtl',inits,metadata={'metrics':metrics.to_dict(),'calibration':profile.metadata(),'objective_weights':objective.__dict__})
            return True

        # Always materialize the initial artifact so zero-epoch smoke runs are useful.
        write_json(out/'initial_signed88_inits.json',design.artifact(base,metrics=base_metrics.to_dict(),extra={'calibration':profile.metadata(),'objective_weights':objective.__dict__}))
        if not (out/'best_signed88_inits.json').exists():
            write_json(out/'best_signed88_inits.json',design.artifact(base,metrics=base_metrics.to_dict(),extra={'stage':'initial','calibration':profile.metadata(),'objective_weights':objective.__dict__,'train_args':vars(args)}))
            design.export_rtl(Path(args.rtl_template_root),out/'best_rtl',base,metadata={'metrics':base_metrics.to_dict(),'calibration':profile.metadata(),'objective_weights':objective.__dict__})

        def train_phase(stage,epochs,*,hard_middle,lr,bit0,bit1,mae0,mae1,mred0,mred1,c_init,c_out,model_ref,opt_ref,extra=None):
            nonlocal global_epoch
            if epochs<=0: return
            for g in opt_ref.param_groups: g['lr']=lr
            for local in range(epochs):
                t=0.0 if epochs<=1 else local/(epochs-1)
                bit_w,mae_w,mred_w=lerp(bit0,bit1,t),lerp(mae0,mae1,t),lerp(mred0,mred1,t)
                main_t=min(1.0,global_epoch/max(total_main-1,1)); tau=lerp(args.er_temperature_start,args.er_temperature_end,main_t)
                opt_ref.zero_grad(set_to_none=True)
                loss,terms=compute_loss(model_ref,batch,c_init=c_init,c_out=c_out,hard_middle=hard_middle,bit_weight=bit_w,mae_weight=mae_w,mred_weight=mred_w,er_temperature=tau,cfg=loss_cfg)
                loss.backward()
                if args.grad_clip>0: torch.nn.utils.clip_grad_norm_(model_ref.parameters(),args.grad_clip)
                opt_ref.step()
                should_eval=(local==epochs-1 or local%max(1,args.eval_every)==0 or global_epoch%max(1,args.eval_every)==0)
                if should_eval:
                    hard=model_ref.hard_inits(); metrics=evaluate_design(design,hard,profile,objective); terms_f={k:float(v.detach().cpu()) for k,v in terms.items()}; loss_f=float(loss.detach().cpu())
                    improved=save_best(hard,metrics,stage,local,loss_f,terms_f)
                    row={'global_epoch':global_epoch,'stage':stage,'local_epoch':local,'loss':loss_f,'terms':terms_f,'metrics':metrics.to_dict(),'improved':improved,'weights':{'bit':bit_w,'mae':mae_w,'mred':mred_w,'er_tau':tau},'extra':extra or {}}
                    with history_path.open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
                    if improved or local%max(1,args.print_every)==0 or local==epochs-1:
                        print(f'[epoch {global_epoch:06d} {stage}:{local:05d}] loss={loss_f:.7f} bit={terms_f["bit"]:.5f} mred={terms_f["mred"]:.5f} er={terms_f["er"]:.5f} mae={terms_f["mae"]:.5f} hard_{metrics.short()}{" *BEST*" if improved else ""}')
                global_epoch+=1

        train_phase('stage1_soft_bit',args.stage1_epochs,hard_middle=False,lr=args.lr,bit0=args.stage1_bit_weight,bit1=args.stage1_bit_weight,mae0=args.stage1_mae_weight,mae1=args.stage1_mae_weight,mred0=args.stage1_mred_weight,mred1=args.stage1_mred_weight,c_init=args.soft_c_init,c_out=args.soft_c_out,model_ref=model,opt_ref=opt)
        train_phase('stage2_soft_signed_ramp',args.stage2_epochs,hard_middle=False,lr=args.lr,bit0=args.stage2_bit_start,bit1=args.stage2_bit_end,mae0=args.stage2_mae_start,mae1=args.stage2_mae_end,mred0=args.stage2_mred_start,mred1=args.stage2_mred_end,c_init=args.soft_c_init,c_out=args.soft_c_out,model_ref=model,opt_ref=opt)
        train_phase('stage3_hard_ste_signed',args.stage3_epochs,hard_middle=True,lr=args.lr*args.stage3_lr_scale,bit0=args.stage3_bit_weight,bit1=args.stage3_bit_weight,mae0=args.stage3_mae_weight,mae1=args.stage3_mae_weight,mred0=args.stage3_mred_weight,mred1=args.stage3_mred_weight,c_init=args.hard_c_init,c_out=args.hard_c_out,model_ref=model,opt_ref=opt)

        population=[]
        if args.population_size>0 and args.population_epochs>0:
            for member in range(args.population_size):
                member_seed=args.seed+100000+member; rng=random.Random(member_seed); set_seed(member_seed)
                start=design.perturb_inits(best['inits'],args.population_flip_p,rng,force_change=True)
                pm=design.build_model(start,args.population_init_conf,args.population_noise_std).to(device); po=torch.optim.Adam(pm.parameters(),lr=args.population_lr)
                soft=min(args.population_soft_epochs,args.population_epochs); hard=max(0,args.population_epochs-soft)
                print(f'[population {member:02d}] seed={member_seed} hamming={hamming(best["inits"],start,design.spec.search_bits)}')
                train_phase(f'pop{member:02d}_soft',soft,hard_middle=False,lr=args.population_lr,bit0=args.stage2_bit_end,bit1=max(0,args.stage2_bit_end*.2),mae0=args.stage2_mae_end,mae1=0,mred0=.8,mred1=1,c_init=args.soft_c_init,c_out=args.soft_c_out,model_ref=pm,opt_ref=po,extra={'population_member':member})
                train_phase(f'pop{member:02d}_hard',hard,hard_middle=True,lr=args.population_lr*args.stage3_lr_scale,bit0=args.stage3_bit_weight,bit1=args.stage3_bit_weight,mae0=args.stage3_mae_weight,mae1=args.stage3_mae_weight,mred0=1,mred1=1,c_init=args.hard_c_init,c_out=args.hard_c_out,model_ref=pm,opt_ref=po,extra={'population_member':member})
                final=pm.hard_inits(); fm=evaluate_design(design,final,profile,objective); population.append({'member':member,'seed':member_seed,'metrics':fm.to_dict(),'inits':final})
        write_json(out/'population_summary.json',{'population':population})
        best_metrics=evaluate_design(design,best['inits'],profile,objective)
        summary={'design':design.spec.name,'design_spec':design.spec.metadata(),'seed':args.seed,'device':str(device),'calibration':profile.metadata(),'objective_weights':objective.__dict__,'initial_metrics':base_metrics.to_dict(),'best_metrics':best_metrics.to_dict(),'best_stage':best['stage'],'best_json':str(out/'best_signed88_inits.json'),'best_rtl':str(out/'best_rtl'),'train_args':vars(args)}
        write_json(out/'summary.json',summary)
        print(f'[best] {best_metrics.short()}')
        print(f'[best-json] {out/"best_signed88_inits.json"}')
        print(f'[best-rtl] {out/"best_rtl"}')
        return 0
    finally:
        sys.stdout.flush(); sys.stdout=old_out; sys.stderr=old_err; log_f.close()

if __name__=='__main__': raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from signed88.common import ObjectiveWeights, read_json
from signed88.data import load_calibration_csv
from signed88.hardware import choices, get_design
from signed88.metrics import evaluate_design

ROOT=Path(__file__).resolve().parent


def parse_args():
    p=argparse.ArgumentParser(description='Verify trained INITs against Python hard model and optional RTL')
    p.add_argument('--inits-json',required=True)
    p.add_argument('--design',default='auto',choices=('auto',)+choices())
    p.add_argument('--calibration-csv',default=None,help='default: inherit artifact calibration')
    p.add_argument('--calibration-weight-column',default=None,choices=['auto','count','p_calib','weight','probability'])
    p.add_argument('--rtl-dir')
    p.add_argument('--run-rtl',action='store_true')
    p.add_argument('--cells-sim',default='/usr/share/yosys/xilinx/cells_sim.v')
    p.add_argument('--iverilog')
    p.add_argument('--vvp')
    p.add_argument('--score-mred-weight',type=float,default=None)
    p.add_argument('--score-er-weight',type=float,default=None)
    p.add_argument('--score-ned-weight',type=float,default=None)
    p.add_argument('--score-bias-weight',type=float,default=None)
    p.add_argument('--score-uniform-mred-weight',type=float,default=None)
    return p.parse_args()


def design_from(obj,requested):
    declared=obj.get('design') or obj.get('design_spec',{}).get('design')
    if requested=='auto':
        if not declared: raise ValueError('untagged JSON: specify --design')
        return get_design(declared)
    d=get_design(requested)
    if declared and get_design(declared).spec.name!=d.spec.name: raise ValueError('design mismatch')
    return d


def brute_force_relation(design,inits):
    low=design.hard_low_numpy(inits).astype(np.int32)
    errors=[]; approx=[]; exacts=[]
    for a_raw in range(256):
        a=a_raw if a_raw<128 else a_raw-256; al=a_raw&63
        for b_raw in range(256):
            b=b_raw if b_raw<128 else b_raw-256; bl=b_raw&63
            exact=a*b; e=int(low[al*64+bl])-al*bl
            exacts.append(exact);errors.append(e);approx.append(exact+e)
    return np.asarray(exacts,np.int32),np.asarray(approx,np.int32),np.asarray(errors,np.int32)


def resolve_tool(explicit,name):
    if explicit: return explicit
    found=shutil.which(name)
    if not found: raise FileNotFoundError(f'{name} not found; pass --{name}')
    return found


def verify_rtl(design,inits,rtl_dir,args):
    rtl_dir=Path(rtl_dir).resolve(); cells=Path(args.cells_sim).resolve()
    if not cells.exists(): raise FileNotFoundError(cells)
    iverilog=resolve_tool(args.iverilog,'iverilog');vvp=resolve_tool(args.vvp,'vvp')
    exact,approx,_=brute_force_relation(design,inits)
    with tempfile.TemporaryDirectory(prefix='verify_signed88_') as td:
        td=Path(td); exp=td/'expected.hex';tb=td/'tb.v';sim=td/'sim.out'
        exp.write_text(''.join(f'{int(x)&0xffff:04x}\n' for x in approx),encoding='ascii')
        tb.write_text(f'''`timescale 1ns/1ps
module tb;
reg signed [7:0] a,b; wire signed [15:0] prod; reg [15:0] expected[0:65535]; integer ia,ib,idx,errors;
s88_top dut(.a(a),.b(b),.prod(prod));
initial begin
  $readmemh("{exp.as_posix()}",expected); idx=0; errors=0;
  for(ia=0;ia<256;ia=ia+1) for(ib=0;ib<256;ib=ib+1) begin
    a=ia; b=ib; #1;
    if(prod!==expected[idx]) begin errors=errors+1; if(errors<=8) $display("FAIL a=%0d b=%0d got=%h exp=%h",$signed(a),$signed(b),prod,expected[idx]); end
    idx=idx+1;
  end
  if(errors!=0) $fatal(1,"RTL mismatches=%0d",errors);
  $display("PASS: all 65536 signed pairs"); $finish;
end endmodule
''',encoding='utf-8')
        sources=sorted(str(p) for p in rtl_dir.glob('*.v'))
        subprocess.run([iverilog,'-g2012','-s','tb','-o',str(sim),str(cells),*sources,str(tb)],check=True)
        subprocess.run([vvp,str(sim)],check=True)


def main():
    args=parse_args();obj=read_json(Path(args.inits_json));design=design_from(obj,args.design);inits=design.normalize_inits(obj.get('inits',obj))
    inherited_cal=obj.get('calibration',{}) if isinstance(obj.get('calibration',{}),dict) else {}
    cal_path=args.calibration_csv or inherited_cal.get('source') or str(ROOT/'data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv')
    if not Path(cal_path).exists(): cal_path=str(ROOT/'data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv')
    cal_col=args.calibration_weight_column or inherited_cal.get('weight_column') or 'auto'
    profile=load_calibration_csv(Path(cal_path),cal_col)
    inherited_obj=obj.get('objective_weights',{}) if isinstance(obj.get('objective_weights',{}),dict) else {}
    objective=ObjectiveWeights(
        inherited_obj.get('workload_mred',1.0) if args.score_mred_weight is None else args.score_mred_weight,
        inherited_obj.get('workload_er',0.25) if args.score_er_weight is None else args.score_er_weight,
        inherited_obj.get('workload_ned',0.10) if args.score_ned_weight is None else args.score_ned_weight,
        inherited_obj.get('workload_bias',0.05) if args.score_bias_weight is None else args.score_bias_weight,
        inherited_obj.get('uniform_mred',0.05) if args.score_uniform_mred_weight is None else args.score_uniform_mred_weight,
    )
    m=evaluate_design(design,inits,profile,objective);print(f'[design] {design.spec.name}');print(f'[metrics] {m.short()}')
    exact,approx,error=brute_force_relation(design,inits)
    assert len(exact)==65536 and np.array_equal(approx-exact,error)
    print('[signed8] PASS: explicitly evaluated all 65,536 signed pairs using final signed16 outputs')
    saved=obj.get('metrics')
    if saved:
        for name in ('ER','MED','MRED','WCE','bias','workload_ER','workload_MED','workload_MRED','workload_WCE','workload_bias'):
            if name not in saved: continue
            a=float(saved[name]);b=float(getattr(m,name));
            if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10): raise AssertionError(f'saved metric mismatch {name}: {a} != {b}')
        print('[artifact] PASS: saved metrics match recomputation')
    if args.rtl_dir:
        art=Path(args.rtl_dir)/'trained_artifact.json'
        if art.exists():
            r=read_json(art);rd=get_design(r['design']);assert rd.spec.name==design.spec.name;assert design.normalize_inits(r['inits'])==inits;print('[rtl-artifact] PASS: patched RTL INIT metadata matches JSON')
    if args.run_rtl:
        if not args.rtl_dir: raise ValueError('--run-rtl requires --rtl-dir')
        verify_rtl(design,inits,args.rtl_dir,args);print('[rtl] PASS')
    return 0
if __name__=='__main__': raise SystemExit(main())

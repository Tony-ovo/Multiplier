#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,shutil
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();root=Path(args.root).resolve();rows=[]
 for path in sorted(root.glob('run_*/summary.json')):
  obj=json.loads(path.read_text());m=obj['best_metrics'];rows.append({'run':path.parent.name,'design':obj['design'],'score':m['objective_score'],'wMRED':m['workload_MRED'],'wER':m['workload_ER'],'wMAE':m['workload_MED'],'uMRED':m['MRED'],'uER':m['ER'],'uMAE':m['MED'],'WCE':m['WCE'],'bias':m['bias'],'best_json':obj['best_json'],'best_rtl':obj['best_rtl']})
 rows.sort(key=lambda r:r['score']);
 with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
  if rows:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (root/'summary.json').write_text(json.dumps({'rows':rows,'best':rows[0] if rows else None},indent=2),encoding='utf-8')
 if rows:
  best=rows[0];shutil.copy2(best['best_json'],root/'overall_best_signed88_inits.json');print(f"[best] {best['run']} score={best['score']:.12g} wMRED={best['wMRED']:.12g}")
 print(root/'summary.csv');return 0
if __name__=='__main__':raise SystemExit(main())

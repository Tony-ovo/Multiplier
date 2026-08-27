#!/usr/bin/env python3
"""Export a trained artifact as a 256x256 int16 product LUT (.npy).

The array is indexed [a + 128, b + 128] and matches the format consumed by
LLM-FPGA/scripts/run_signed_w8a8_ppl_probe.py.  Name the file
s88ref_<tag>_signed_int8_lut.npy and place it in LLM-FPGA/outputs/fpga_luts/
so the probe picks it up automatically (design id: s88ref_<tag>).

The table is built from the same hard numpy model that verify.py proves
bit-exact against the exported RTL over all 65,536 signed pairs, so the LUT
is guaranteed to match deployed hardware behaviour.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from signed88.common import read_json
from signed88.hardware import choices, get_design


def build_table(design, inits) -> np.ndarray:
    low = design.hard_low_numpy(inits).astype(np.int64)
    a = np.arange(-128, 128, dtype=np.int64).reshape(256, 1)
    b = np.arange(-128, 128, dtype=np.int64).reshape(1, 256)
    al = a & 63
    bl = b & 63
    approx = a * b + low[al * 64 + bl] - al * bl
    if approx.min() < np.iinfo(np.int16).min or approx.max() > np.iinfo(np.int16).max:
        raise ValueError('product table exceeds int16 range')
    return approx.astype(np.int16)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inits-json', required=True)
    p.add_argument('--design', default='auto', choices=('auto',) + choices())
    p.add_argument('--out', required=True)
    args = p.parse_args()
    obj = read_json(Path(args.inits_json))
    declared = obj.get('design') or obj.get('design_spec', {}).get('design')
    name = declared if args.design == 'auto' else args.design
    if not name:
        raise SystemExit('untagged JSON: specify --design')
    design = get_design(name)
    inits = design.normalize_inits(obj.get('inits', obj))
    table = build_table(design, inits)
    np.save(args.out, table)
    exact = np.arange(-128, 128, dtype=np.int64).reshape(256, 1) * np.arange(-128, 128, dtype=np.int64).reshape(1, 256)
    err = table.astype(np.int64) - exact
    print(f'wrote {args.out} shape={table.shape} dtype={table.dtype}')
    print(f'design={design.spec.name} ER={np.mean(err != 0):.6f} MAE={np.abs(err).mean():.4f} WCE={np.abs(err).max()} bias={err.mean():+.4f}')


if __name__ == '__main__':
    main()

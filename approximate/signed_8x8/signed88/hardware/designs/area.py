from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from ...common import hex_to_int
from ...lut import TrainableLUT6_2, TrainableLUT6, int_bits_t, lut62_np
from ..base import BaseDesign, DesignSpec, Low6Core, RtlBinding
from .common import BooleanCoreMixin

BASE={'q16_high45':"64'h066AACC00AA00AA0",'q16_high67':"64'hC8800000A44CC000"}
ALL={k:tuple(range(64)) for k in BASE}
AREA_CSA0=int('96969696e8e8e8e8',16); AREA_CSA=int('69966996e8e8e8e8',16); AREA_TAIL=int('00ffff00d0d0d0d0',16)

class AreaCore(Low6Core,BooleanCoreMixin):
    def __init__(self,inits:Mapping[str,str],mutable_bits,init_conf,noise_std):
        super().__init__(); self.tables=nn.ModuleDict({n:TrainableLUT6_2(inits[n],mutable_bits[n],init_conf,noise_std) for n in BASE})
        from ...common import int_bits
        self.register_buffer('area_csa0',torch.tensor(int_bits(AREA_CSA0),dtype=torch.float32)); self.register_buffer('area_csa',torch.tensor(int_bits(AREA_CSA),dtype=torch.float32)); self.register_buffer('area_tail',torch.tensor(int_bits(AREA_TAIL),dtype=torch.float32))
    def child(self,a,digit,*,c_init,c_out,hard_middle):
        ab=int_bits_t(a,6); db=int_bits_t(digit,2); one=torch.ones_like(ab[0]); z=torch.zeros_like(ab[0])
        p4,p5=self.tables['q16_high45'](db[0],db[1],ab[3],ab[4],ab[5],one,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        p6,p7=self.tables['q16_high67'](db[0],db[1],ab[3],ab[4],ab[5],one,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        return [z,z,z,z,p4,p5,p6,p7]
    def forward_bits(self,al,bl,*,c_init,c_out,hard_middle):
        pl=self.child(al,bl&3,c_init=c_init,c_out=c_out,hard_middle=hard_middle); pm=self.child(al,(bl>>2)&3,c_init=c_init,c_out=c_out,hard_middle=hard_middle); ph=self.child(al,(bl>>4)&3,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        z=torch.zeros_like(pl[0]); o=torch.ones_like(pl[0]); a_row=ph; b_row=pm[2:8]; c_row=pl[4:8]+[z,z]; p=[z for _ in range(8)]; g=[z for _ in range(8)]; p[7]=a_row[7]
        g[0],p[0]=self.fixed_lut62(self.area_csa0,[c_row[0],b_row[0],a_row[0],o,o,o],c_out=c_out,hard_middle=hard_middle)
        for i in range(1,6): g[i],p[i]=self.fixed_lut62(self.area_csa,[c_row[i],b_row[i],a_row[i],g[i-1],o,o],c_out=c_out,hard_middle=hard_middle)
        _,p[6]=self.fixed_lut62(self.area_tail,[z,z,z,a_row[6],g[5],o],c_out=c_out,hard_middle=hard_middle)
        upper=[]; carry=z
        for bit in range(4,8):
            s=p[bit]; di=g[bit-1]; out=self.xor2(s,carry); next_c=s*carry+(1-s)*di; out=self.fixed_node(out,c_out=c_out,hard_middle=hard_middle); carry=self.fixed_node(next_c,c_out=c_out,hard_middle=hard_middle); upper.append(out)
        return [z,z,z,z]+p[:4]+upper
    def hard_inits(self): return {n:self.tables[n].hard_hex() for n in BASE}
    def bin_reg(self): return torch.stack([m.bin_reg() for m in self.tables.values()]).mean()

class AreaDesign(BaseDesign):
    def __init__(self):
        self.spec=DesignSpec(name='area',rtl_dir='Area',resource_summary='29 LUT6_2 + 5 CARRY4',base_inits=BASE,mutable_bits=ALL,search_bits=ALL,rtl_bindings=(RtlBinding('q16_high45','s8862_approx62_q16.v','s8862_approx62_q16','high45_lut'),RtlBinding('q16_high67','s8862_approx62_q16.v','s8862_approx62_q16','high67_lut')))
    def build_core(self,inits,init_conf,noise_std): return AreaCore(inits,self.spec.mutable_bits,init_conf,noise_std)
    @staticmethod
    def _child_np(ints,a,d):
        ab=[((a>>i)&1).astype(np.uint64) for i in range(6)]; d0=(d&1).astype(np.uint64);d1=((d>>1)&1).astype(np.uint64);addr=d0+(d1<<1)+(ab[3]<<2)+(ab[4]<<3)+(ab[5]<<4)+np.uint64(32);p4,p5=lut62_np(ints['q16_high45'],addr);p6,p7=lut62_np(ints['q16_high67'],addr);bits=[np.zeros_like(a,dtype=np.uint16) for _ in range(4)]+[p4,p5,p6,p7];return bits
    def hard_low_numpy(self,inits):
        ints={k:hex_to_int(v) for k,v in self.normalize_inits(inits).items()};a=np.repeat(np.arange(64,dtype=np.uint16),64);b=np.tile(np.arange(64,dtype=np.uint16),64);pl=self._child_np(ints,a,b&3);pm=self._child_np(ints,a,(b>>2)&3);ph=self._child_np(ints,a,(b>>4)&3)
        z=np.zeros_like(a,dtype=np.uint16);a_row=ph;b_row=pm[2:8];c_row=pl[4:8]+[z,z];p=[z.copy() for _ in range(8)];g=[z.copy() for _ in range(8)];p[7]=a_row[7]
        addr=c_row[0].astype(np.uint64)+(b_row[0].astype(np.uint64)<<1)+(a_row[0].astype(np.uint64)<<2)+np.uint64(8+16+32);g[0],p[0]=lut62_np(AREA_CSA0,addr)
        for i in range(1,6):
            addr=c_row[i].astype(np.uint64)+(b_row[i].astype(np.uint64)<<1)+(a_row[i].astype(np.uint64)<<2)+(g[i-1].astype(np.uint64)<<3)+np.uint64(16+32);g[i],p[i]=lut62_np(AREA_CSA,addr)
        addr=(a_row[6].astype(np.uint64)<<3)+(g[5].astype(np.uint64)<<4)+np.uint64(32);_,p[6]=lut62_np(AREA_TAIL,addr)
        upper=[];carry=z
        for bit in range(4,8):
            s=p[bit];di=g[bit-1];upper.append((s^carry).astype(np.uint16));carry=np.where(s!=0,carry,di).astype(np.uint16)
        bits=[z,z,z,z]+p[:4]+upper;value=np.zeros_like(a,dtype=np.int32)
        for i,x in enumerate(bits):value+=x.astype(np.int32)<<i
        return value

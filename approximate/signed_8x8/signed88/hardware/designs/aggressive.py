from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from ...common import hex_to_int
from ...lut import TrainableLUT6, TrainableLUT6_2, int_bits_t, lut62_np, lut_bit_np
from ..base import BaseDesign, DesignSpec, Low6Core, RtlBinding

EDGE_REACH = tuple(list(range(16,32)) + list(range(48,64)))
COMP_EDGE_REACH = tuple(list(range(0,16)) + list(range(32,48)))
ALL = tuple(range(64))

BASE = {
 'low_lut01': "64'hEAC0D38BA0A02517", 'low_lut23': "64'hEEAACC00EAC0EAC0", 'low_lut45': "64'hEE22CC00EA40EAC0", 'low_lut67': "64'hC80039B3C4001018",
 'mid_lut01': "64'hEAC0D38BA0A02517", 'mid_lut23': "64'hEE22CC00E240EAC0", 'mid_lut45': "64'hACAA4C80A8C0EA40", 'mid_lut67': "64'hC80039B32C801018",
 'high_lut01': "64'hEAC0D38BA0A02517", 'high_lut23': "64'hEE22CC00E240EAC0", 'high_lut45': "64'hF02A4D898862E6C0", 'high_lut67': "64'hC80039B324801018",
 'comp23': "64'h42F1FFF85D55FEE6", 'comp4': "64'hFFFFFFFEFFFEFE96", 'comp5': "64'hFFFFFFFFFFFFFFE8", 'comp6': "64'hFDFF75FEFF7E5696", 'comp7': "64'hFDFFDDFFDFBF55E8", 'comp89': "64'hC20BDCC81222C8F6",
}
MUTABLE = {}
for name in BASE:
    if name.endswith('lut01') or name.endswith('lut67'):
        MUTABLE[name] = EDGE_REACH
    elif name in ('comp23','comp89'):
        MUTABLE[name] = COMP_EDGE_REACH
    else:
        MUTABLE[name] = ALL


class AggressiveCore(Low6Core):
    def __init__(self, inits: Mapping[str,str], mutable_bits, init_conf: float, noise_std: float):
        super().__init__()
        lut62_names = [n for n in BASE if n.startswith(('low_','mid_','high_'))] + ['comp23','comp89']
        lut6_names = ['comp4','comp5','comp6','comp7']
        self.lut62 = nn.ModuleDict({n: TrainableLUT6_2(inits[n], mutable_bits[n], init_conf, noise_std) for n in lut62_names})
        self.lut6 = nn.ModuleDict({n: TrainableLUT6(inits[n], mutable_bits[n], init_conf, noise_std) for n in lut6_names})

    def child(self, prefix: str, a, digit, *, c_init, c_out, hard_middle):
        ab = int_bits_t(a,6); db=int_bits_t(digit,2); one=torch.ones_like(ab[0])
        p0,p1=self.lut62[f'{prefix}_lut01'](db[0],db[1],ab[0],ab[1],one,one,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        p2,p3=self.lut62[f'{prefix}_lut23'](db[0],db[1],ab[1],ab[2],ab[3],one,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        p4,p5=self.lut62[f'{prefix}_lut45'](db[0],db[1],ab[3],ab[4],ab[5],one,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        p6,p7=self.lut62[f'{prefix}_lut67'](db[0],db[1],ab[4],ab[5],one,one,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        return [p0,p1,p2,p3,p4,p5,p6,p7]

    def forward_bits(self, al, bl, *, c_init, c_out, hard_middle):
        plow=self.child('low',al,bl&3,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        pmid=self.child('mid',al,(bl>>2)&3,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        phigh=self.child('high',al,(bl>>4)&3,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        z=torch.zeros_like(plow[0]); o=torch.ones_like(plow[0]); prod=[z for _ in range(12)]
        prod[0],prod[1]=plow[0],plow[1]; prod[10],prod[11]=phigh[6],phigh[7]
        prod[2],prod[3]=self.lut62['comp23'](plow[2],pmid[0],plow[3],pmid[1],z,o,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        addr4=(plow[4],pmid[2],phigh[0],plow[5],pmid[3],phigh[1])
        addr6=(plow[6],pmid[4],phigh[2],plow[7],pmid[5],phigh[3])
        prod[4]=self.lut6['comp4'](*addr4,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        prod[5]=self.lut6['comp5'](*addr4,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        prod[6]=self.lut6['comp6'](*addr6,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        prod[7]=self.lut6['comp7'](*addr6,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        prod[8],prod[9]=self.lut62['comp89'](pmid[6],phigh[4],pmid[7],phigh[5],z,o,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        return prod

    def hard_inits(self):
        out={n:m.hard_hex() for n,m in self.lut62.items()}; out.update({n:m.hard_hex() for n,m in self.lut6.items()}); return {n:out[n] for n in BASE}
    def bin_reg(self):
        regs=[m.bin_reg() for m in list(self.lut62.values())+list(self.lut6.values())]; return torch.stack(regs).mean()


class AggressiveDesign(BaseDesign):
    def __init__(self):
        bindings=[]
        for prefix,module in [('low','s8862_aggr62_low'),('mid','s8862_aggr62_mid'),('high','s8862_aggr62_high')]:
            for role,inst in [('lut01','lut01'),('lut23','lut23'),('lut45','lut45'),('lut67','lut67')]:
                bindings.append(RtlBinding(f'{prefix}_{role}','s8862_approx66_aggressive.v',module,inst))
        for table,inst in [('comp23','comp23_lut'),('comp4','comp4_lut'),('comp5','comp5_lut'),('comp6','comp6_lut'),('comp7','comp7_lut'),('comp89','comp89_lut')]:
            bindings.append(RtlBinding(table,'s8862_approx66_aggressive.v','s8862_aggr_comp66',inst))
        self.spec=DesignSpec(name='aggressive',rtl_dir='Aggressive',resource_summary='31 LUT6_2 + 4 LUT6 + 4 CARRY4',base_inits=BASE,mutable_bits=MUTABLE,search_bits=MUTABLE,rtl_bindings=tuple(bindings))
    def build_core(self,inits,init_conf,noise_std): return AggressiveCore(inits,self.spec.mutable_bits,init_conf,noise_std)

    @staticmethod
    def _child_np(ints,prefix,a,digit):
        ab=[((a>>i)&1).astype(np.uint64) for i in range(6)]; d0=(digit&1).astype(np.uint64); d1=((digit>>1)&1).astype(np.uint64)
        addr01=d0+(d1<<1)+(ab[0]<<2)+(ab[1]<<3)+np.uint64(16+32)
        addr23=d0+(d1<<1)+(ab[1]<<2)+(ab[2]<<3)+(ab[3]<<4)+np.uint64(32)
        addr45=d0+(d1<<1)+(ab[3]<<2)+(ab[4]<<3)+(ab[5]<<4)+np.uint64(32)
        addr67=d0+(d1<<1)+(ab[4]<<2)+(ab[5]<<3)+np.uint64(16+32)
        pairs=[lut62_np(ints[f'{prefix}_lut01'],addr01),lut62_np(ints[f'{prefix}_lut23'],addr23),lut62_np(ints[f'{prefix}_lut45'],addr45),lut62_np(ints[f'{prefix}_lut67'],addr67)]
        bits=[x for pair in pairs for x in pair]; value=np.zeros_like(a,dtype=np.int32)
        for i,p in enumerate(bits): value += p.astype(np.int32)<<i
        return bits,value

    def hard_low_numpy(self,inits):
        ints={k:hex_to_int(v) for k,v in self.normalize_inits(inits).items()}; a=np.repeat(np.arange(64,dtype=np.uint16),64); b=np.tile(np.arange(64,dtype=np.uint16),64)
        pl,pv=self._child_np(ints,'low',a,b&3); pm,mv=self._child_np(ints,'mid',a,(b>>2)&3); ph,hv=self._child_np(ints,'high',a,(b>>4)&3)
        prod=[np.zeros_like(a,dtype=np.uint16) for _ in range(12)]; prod[0]=pl[0];prod[1]=pl[1];prod[10]=ph[6];prod[11]=ph[7]
        addr23=pl[2].astype(np.uint64)+(pm[0].astype(np.uint64)<<1)+(pl[3].astype(np.uint64)<<2)+(pm[1].astype(np.uint64)<<3)+np.uint64(32)
        prod[2],prod[3]=lut62_np(ints['comp23'],addr23)
        addr4=pl[4].astype(np.uint64)+(pm[2].astype(np.uint64)<<1)+(ph[0].astype(np.uint64)<<2)+(pl[5].astype(np.uint64)<<3)+(pm[3].astype(np.uint64)<<4)+(ph[1].astype(np.uint64)<<5)
        addr6=pl[6].astype(np.uint64)+(pm[4].astype(np.uint64)<<1)+(ph[2].astype(np.uint64)<<2)+(pl[7].astype(np.uint64)<<3)+(pm[5].astype(np.uint64)<<4)+(ph[3].astype(np.uint64)<<5)
        for i,name,addr in [(4,'comp4',addr4),(5,'comp5',addr4),(6,'comp6',addr6),(7,'comp7',addr6)]: prod[i]=lut_bit_np(ints[name],addr)
        addr89=pm[6].astype(np.uint64)+(ph[4].astype(np.uint64)<<1)+(pm[7].astype(np.uint64)<<2)+(ph[5].astype(np.uint64)<<3)+np.uint64(32)
        prod[8],prod[9]=lut62_np(ints['comp89'],addr89)
        value=np.zeros_like(a,dtype=np.int32)
        for i,p in enumerate(prod): value += p.astype(np.int32)<<i
        return value

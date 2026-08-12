from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .common import MAX_ABS_PRODUCT
from .data import TorchCalibration, UNIFORM_NONZERO_PROB, UNIFORM_RED_WEIGHT, UNIFORM_STATE_PROB, UNIFORM_ZERO_PROB

ERROR_SCALE = 63.0 * 63.0


@dataclass(frozen=True)
class LossConfig:
    calibration_mix: float = 0.98
    er_weight: float = 0.25
    bias_weight: float = 0.05
    zero_weight: float = 0.25
    symmetry_weight: float = 0.0
    bin_weight: float = 0.0
    bit_weighting: str = 'linear'


def bit_weight_vector(mode: str, device: torch.device) -> torch.Tensor:
    if mode == 'uniform': w=torch.ones(12,device=device)
    elif mode == 'linear': w=torch.linspace(1.0,2.0,12,device=device)
    elif mode == 'sqrt_value': w=torch.sqrt(torch.tensor([float(1<<i) for i in range(12)],device=device))
    elif mode == 'value': w=torch.tensor([float(1<<i) for i in range(12)],device=device)
    else: raise ValueError(mode)
    return w/w.mean()


def _mix(workload, uniform, alpha: float):
    return float(alpha)*workload + (1.0-float(alpha))*uniform


def compute_loss(
    model,
    batch: TorchCalibration,
    *,
    c_init: float,
    c_out: float,
    hard_middle: bool,
    bit_weight: float,
    mae_weight: float,
    mred_weight: float,
    er_temperature: float,
    cfg: LossConfig,
):
    approx_signed, low_value, low_bits = model.forward_signed_rows(
        batch.a,batch.b,batch.state_index,c_init=c_init,c_out=c_out,hard_middle=hard_middle
    )
    signed_error = approx_signed - batch.exact
    signed_ed = torch.abs(signed_error)
    exact_ll = model.grid_exact_ll
    low_error = low_value - exact_ll
    low_ed = torch.abs(low_error)

    # Final signed-int8 workload losses: these operate on the CSV rows directly.
    w_mae = torch.sum(batch.probability * signed_ed) / ERROR_SCALE
    nonzero = batch.exact != 0
    w_mred = torch.sum(batch.probability[nonzero] * signed_ed[nonzero] / torch.abs(batch.exact[nonzero])) / max(batch.nonzero_probability,1e-12)
    w_er = torch.sum(batch.probability * (1.0 - torch.exp(-signed_ed / max(float(er_temperature),1e-6))))
    w_bias = torch.abs(torch.sum(batch.probability * signed_error)) / ERROR_SCALE
    zero = batch.exact == 0
    if bool(torch.any(zero)):
        w_zero = torch.sum(batch.probability[zero] * signed_ed[zero]) / max(batch.zero_probability,1e-12) / ERROR_SCALE
    else:
        w_zero = torch.zeros((),device=signed_ed.device)

    # Uniform signed-int8 safety losses are exactly folded onto the 4096 LL states.
    u_state = torch.as_tensor(UNIFORM_STATE_PROB,dtype=torch.float32,device=low_value.device)
    u_red = torch.as_tensor(UNIFORM_RED_WEIGHT,dtype=torch.float32,device=low_value.device)
    u_zero_p = torch.as_tensor(UNIFORM_ZERO_PROB,dtype=torch.float32,device=low_value.device)
    u_mae = torch.sum(u_state * low_ed) / ERROR_SCALE
    u_mred = torch.sum(u_red * low_ed) / max(UNIFORM_NONZERO_PROB,1e-12)
    u_er = torch.sum(u_state * (1.0 - torch.exp(-low_ed / max(float(er_temperature),1e-6))))
    u_bias = torch.abs(torch.sum(u_state * low_error)) / ERROR_SCALE
    u_zero = torch.sum(u_zero_p * low_ed) / max(float(UNIFORM_ZERO_PROB.sum()),1e-12) / ERROR_SCALE

    mae = _mix(w_mae,u_mae,cfg.calibration_mix)
    mred = _mix(w_mred,u_mred,cfg.calibration_mix)
    er = _mix(w_er,u_er,cfg.calibration_mix)
    bias = _mix(w_bias,u_bias,cfg.calibration_mix)
    zero_loss = _mix(w_zero,u_zero,cfg.calibration_mix)

    # Auxiliary low-product bit supervision. It helps early optimization but the
    # principal numerical losses above are always on final signed8x8 results.
    exact_ll_i = (model.grid_al * model.grid_bl).to(torch.int64)
    exact_bits = torch.stack([((exact_ll_i>>i)&1).to(torch.float32) for i in range(12)],dim=0)
    pred_bits = torch.stack(low_bits,dim=0)
    state_w = cfg.calibration_mix*batch.state_probability + (1-cfg.calibration_mix)*u_state
    bw = bit_weight_vector(cfg.bit_weighting, low_value.device)
    if hard_middle:
        per_point = torch.abs(pred_bits-exact_bits)
    else:
        per_point = F.binary_cross_entropy(torch.clamp(pred_bits,1e-6,1-1e-6),exact_bits,reduction='none')
    per_bit = torch.sum(per_point*state_w.unsqueeze(0),dim=1)
    bit_loss = torch.sum(per_bit*bw)/torch.sum(bw)

    matrix=low_value.reshape(64,64)
    symmetry=torch.mean(torch.abs(matrix-matrix.T))/ERROR_SCALE
    bin_reg=model.bin_reg()
    total=(
        float(bit_weight)*bit_loss + float(mae_weight)*mae + float(mred_weight)*mred
        + cfg.er_weight*er + cfg.bias_weight*bias + cfg.zero_weight*zero_loss
        + cfg.symmetry_weight*symmetry + cfg.bin_weight*bin_reg
    )
    terms={
        'bit':bit_loss,'mae':mae,'mred':mred,'er':er,'bias':bias,'zero':zero_loss,
        'symmetry':symmetry,'bin':bin_reg,'workload_mae':w_mae,'workload_mred':w_mred,
        'workload_er_surrogate':w_er,'uniform_mred':u_mred,
    }
    return total, terms

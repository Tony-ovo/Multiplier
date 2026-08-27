from __future__ import annotations

import math
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
    er_weight: float = 0.0001
    mse_weight: float = 1.0
    bias_weight: float = 1.0
    conditional_bias_weight: float = 0.001
    bias_effective_k: float = 1024.0
    zero_weight: float = 0.25
    symmetry_weight: float = 0.0
    bin_weight: float = 0.0
    bit_weighting: str = 'linear'
    # Differentiable worst-case-error surrogate over the full 4096 LL states:
    # logsumexp(beta*|e|)/beta, normalized by ERROR_SCALE.  Mirrors the
    # discrete uniform_wce objective term used by the hard checkpoint selector.
    wce_weight: float = 0.0
    wce_beta: float = 0.25


def _validate_loss_config(cfg: LossConfig) -> None:
    if not 0.0 <= float(cfg.calibration_mix) <= 1.0:
        raise ValueError('calibration_mix must be in [0,1]')
    if not math.isfinite(float(cfg.bias_effective_k)) or float(cfg.bias_effective_k) < 1.0:
        raise ValueError('bias_effective_k must be finite and >= 1')
    for name in (
        'er_weight', 'mse_weight', 'bias_weight', 'conditional_bias_weight',
        'zero_weight', 'symmetry_weight', 'bin_weight', 'wce_weight',
    ):
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'{name} must be finite and non-negative')
    beta = float(cfg.wce_beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError('wce_beta must be finite and > 0')


def bit_weight_vector(mode: str, device: torch.device) -> torch.Tensor:
    if mode == 'uniform': w=torch.ones(12,device=device)
    elif mode == 'linear': w=torch.linspace(1.0,2.0,12,device=device)
    elif mode == 'sqrt_value': w=torch.sqrt(torch.tensor([float(1<<i) for i in range(12)],device=device))
    elif mode == 'value': w=torch.tensor([float(1<<i) for i in range(12)],device=device)
    else: raise ValueError(mode)
    return w/w.mean()


def _mix(workload, uniform, alpha: float):
    return float(alpha)*workload + (1.0-float(alpha))*uniform


def _conditional_mean_square(
    error: torch.Tensor,
    probability: torch.Tensor,
    group_index: torch.Tensor,
    group_count: int = 256,
) -> torch.Tensor:
    """Return sum_g P(g) * E[error|g]^2 without losing rare groups."""
    mass = torch.zeros(group_count, dtype=error.dtype, device=error.device)
    weighted_error = torch.zeros_like(mass)
    mass.scatter_add_(0, group_index, probability)
    weighted_error.scatter_add_(0, group_index, probability * error)
    mean = weighted_error / torch.clamp(mass, min=1e-12)
    return torch.sum(mass * torch.square(mean))


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
    gemm_weight: float = 1.0,
):
    _validate_loss_config(cfg)
    if not math.isfinite(float(gemm_weight)) or float(gemm_weight) < 0.0:
        raise ValueError('gemm_weight must be finite and non-negative')
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
    w_mse_raw = torch.sum(batch.probability * torch.square(signed_error))
    w_signal_energy = torch.clamp(
        torch.sum(batch.probability * torch.square(batch.exact)), min=1.0
    )
    w_nmse = w_mse_raw / w_signal_energy
    nonzero = batch.exact != 0
    w_mred = torch.sum(batch.probability[nonzero] * signed_ed[nonzero] / torch.abs(batch.exact[nonzero])) / max(batch.nonzero_probability,1e-12)
    w_er = torch.sum(batch.probability * (1.0 - torch.exp(-signed_ed / max(float(er_temperature),1e-6))))
    w_bias_signed = torch.sum(batch.probability * signed_error)
    w_bias_squared = torch.square(w_bias_signed) / w_signal_energy
    w_cond_a_raw = _conditional_mean_square(
        signed_error, batch.probability, (batch.a + 128).to(torch.long)
    )
    w_cond_b_raw = _conditional_mean_square(
        signed_error, batch.probability, (batch.b + 128).to(torch.long)
    )
    w_cond_excess = torch.clamp(
        0.5 * (w_cond_a_raw + w_cond_b_raw) - torch.square(w_bias_signed), min=0.0
    ) / w_signal_energy
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
    u_mse_raw = torch.sum(u_state * torch.square(low_error))
    signed_values = torch.arange(-128, 128, dtype=low_value.dtype, device=low_value.device)
    u_signal_energy = torch.clamp(torch.mean(torch.square(signed_values)) ** 2, min=1.0)
    u_nmse = u_mse_raw / u_signal_energy
    u_mred = torch.sum(u_red * low_ed) / max(UNIFORM_NONZERO_PROB,1e-12)
    u_er = torch.sum(u_state * (1.0 - torch.exp(-low_ed / max(float(er_temperature),1e-6))))
    u_bias_signed = torch.sum(u_state * low_error)
    u_bias_squared = torch.square(u_bias_signed) / u_signal_energy
    u_matrix = low_error.reshape(64, 64)
    u_cond_a_raw = torch.mean(torch.square(torch.mean(u_matrix, dim=1)))
    u_cond_b_raw = torch.mean(torch.square(torch.mean(u_matrix, dim=0)))
    u_cond_excess = torch.clamp(
        0.5 * (u_cond_a_raw + u_cond_b_raw) - torch.square(u_bias_signed), min=0.0
    ) / u_signal_energy
    u_zero = torch.sum(u_zero_p * low_ed) / max(float(UNIFORM_ZERO_PROB.sum()),1e-12) / ERROR_SCALE

    mae = _mix(w_mae,u_mae,cfg.calibration_mix)
    # The GEMM objective follows the measured AI workload exactly.  Uniform
    # signed-int8 data remains an auxiliary safety regularizer through the
    # legacy MAE/MRED/ER/zero terms; it must not silently change the primary
    # MSE/bias objective used by the hard checkpoint selector.
    nmse = w_nmse
    mred = _mix(w_mred,u_mred,cfg.calibration_mix)
    er = _mix(w_er,u_er,cfg.calibration_mix)
    k_minus_one = max(float(cfg.bias_effective_k) - 1.0, 0.0)
    bias = k_minus_one * w_bias_squared
    conditional_bias = k_minus_one * w_cond_excess
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
    # Smooth worst-case proxy over all 4096 uniform LL states.  logsumexp is a
    # tight, numerically stable upper bound of the max whose gradient spreads
    # over the current worst offenders instead of a single argmax point.
    wce_soft = torch.logsumexp(low_ed * float(cfg.wce_beta), dim=0) / float(cfg.wce_beta) / ERROR_SCALE
    total=(
        float(bit_weight)*bit_loss + float(mae_weight)*mae + float(mred_weight)*mred
        + float(gemm_weight) * (
            cfg.mse_weight*nmse + cfg.bias_weight*bias
            + cfg.conditional_bias_weight*conditional_bias
        )
        + cfg.er_weight*er + cfg.zero_weight*zero_loss
        + cfg.symmetry_weight*symmetry + cfg.bin_weight*bin_reg
        + cfg.wce_weight*wce_soft
    )
    terms={
        'bit':bit_loss,'mae':mae,'nmse':nmse,'mred':mred,'er':er,'wce':wce_soft,
        'gemm_weight':torch.as_tensor(float(gemm_weight),device=nmse.device),
        'bias':bias,'conditional_bias':conditional_bias,'zero':zero_loss,
        'symmetry':symmetry,'bin':bin_reg,'workload_mae':w_mae,'workload_mred':w_mred,
        'workload_nmse':w_nmse,'workload_mse':w_mse_raw,
        'workload_signed_bias':w_bias_signed,
        'workload_bias_squared':w_bias_squared,
        'workload_conditional_bias_a_rms':torch.sqrt(torch.clamp_min(w_cond_a_raw,0.0)),
        'workload_conditional_bias_b_rms':torch.sqrt(torch.clamp_min(w_cond_b_raw,0.0)),
        'workload_conditional_bias_excess':w_cond_excess,
        'workload_gemm_nmse':w_nmse + k_minus_one * (
            w_bias_squared + cfg.conditional_bias_weight * w_cond_excess
        ),
        'workload_er_surrogate':w_er,'uniform_mred':u_mred,'uniform_nmse':u_nmse,
    }
    return total, terms

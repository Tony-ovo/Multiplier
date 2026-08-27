from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .common import (
    LL_STATES, MAX_ABS_PRODUCT, Metrics, ObjectiveWeights, SIGNED_TOTAL,
    validate_objective_weights,
)
from .data import CalibrationProfile, UNIFORM_NONZERO_PROB, UNIFORM_RED_WEIGHT, UNIFORM_ZERO_PROB


def _conditional_mean_square(error, probability, values):
    group = values.astype(np.int64) + 128
    mass = np.bincount(group, weights=probability, minlength=256)
    weighted_error = np.bincount(
        group, weights=probability * error.astype(np.float64), minlength=256
    )
    mean = np.divide(
        weighted_error, mass, out=np.zeros_like(weighted_error), where=mass > 0.0
    )
    return float(np.sum(mass * np.square(mean)))


def evaluate_design(design, inits: Mapping[str,str], profile: CalibrationProfile, objective: ObjectiveWeights) -> Metrics:
    validate_objective_weights(objective)
    low = design.hard_low_numpy(inits).astype(np.int32)
    if low.shape != (LL_STATES,):
        raise ValueError(f'{design.spec.name}: hard_low_numpy returned {low.shape}')
    al = np.repeat(np.arange(64,dtype=np.int32),64)
    bl = np.tile(np.arange(64,dtype=np.int32),64)
    exact_ll = al * bl
    error = low - exact_ll
    ed = np.abs(error).astype(np.int64)
    mask = ed != 0

    error_cases = int(np.count_nonzero(mask)) * 16
    med = float(ed.mean())
    uniform_mse = float(np.mean(np.square(error.astype(np.float64))))
    signed_values = np.arange(-128,128,dtype=np.float64)
    uniform_signal_energy = max(float(np.mean(np.square(signed_values)) ** 2), 1.0)
    uniform_nmse = uniform_mse / uniform_signal_energy
    uniform_mred_total = float(np.sum(ed.astype(np.float64) * UNIFORM_RED_WEIGHT))
    uniform_mred = uniform_mred_total / max(UNIFORM_NONZERO_PROB, 1e-15)
    zero_violations = int(round(float(np.sum(UNIFORM_ZERO_PROB[mask])) * SIGNED_TOTAL))
    low_matrix = low.reshape(64,64)
    symmetry_violations = int(np.count_nonzero(low_matrix != low_matrix.T)) * 16

    row_error = error[profile.state_index]
    row_ed = np.abs(row_error).astype(np.int64)
    p = profile.probability
    row_mask = row_ed != 0
    workload_er = float(np.sum(p[row_mask]))
    workload_med = float(np.sum(p * row_ed))
    nonzero = profile.exact != 0
    workload_mred_total = float(np.sum(p[nonzero] * row_ed[nonzero] / np.abs(profile.exact[nonzero])))
    workload_mred = workload_mred_total / max(profile.nonzero_probability,1e-15)
    workload_bias = float(np.sum(p * row_error))
    workload_mse = float(np.sum(p * np.square(row_error.astype(np.float64))))
    workload_rmse = math.sqrt(workload_mse)
    workload_signal_energy = max(
        float(np.sum(p * np.square(profile.exact.astype(np.float64)))), 1.0
    )
    workload_nmse = workload_mse / workload_signal_energy
    conditional_a_squared = _conditional_mean_square(row_error,p,profile.a)
    conditional_b_squared = _conditional_mean_square(row_error,p,profile.b)
    conditional_excess_raw = max(
        0.0, 0.5 * (conditional_a_squared + conditional_b_squared) - workload_bias ** 2
    )
    normalized_bias_squared = workload_bias ** 2 / workload_signal_energy
    conditional_excess_normalized = conditional_excess_raw / workload_signal_energy
    k = max(float(objective.bias_effective_k),1.0)
    k_minus_one = k - 1.0
    bias_penalty = k_minus_one * normalized_bias_squared
    conditional_bias_penalty = k_minus_one * conditional_excess_normalized
    workload_gemm_nmse = (
        workload_nmse + bias_penalty
        + float(objective.workload_conditional_bias) * conditional_bias_penalty
    )
    predicted_dot_rmse = math.sqrt(
        max(0.0,k*workload_mse + k*k_minus_one*workload_bias**2)
    )
    bias_drift_sigma = (
        math.sqrt(k)*workload_bias/max(workload_rmse,1e-15)
        if workload_rmse > 0.0 else 0.0
    )
    workload_wce = int(row_ed.max()) if len(row_ed) else 0
    zero = profile.exact == 0
    zero_v_prob = float(np.sum(p[zero & row_mask]))
    zero_v_rate = zero_v_prob / max(profile.zero_probability,1e-15)
    workload_ned = workload_med / MAX_ABS_PRODUCT

    uniform_wce = int(ed.max())
    score = (
        objective.workload_nmse * workload_nmse
        + objective.workload_bias * bias_penalty
        + objective.workload_conditional_bias * conditional_bias_penalty
        + objective.workload_mred * workload_mred
        + objective.workload_er * workload_er
        + objective.workload_ned * workload_ned
        + objective.uniform_mred * uniform_mred
        + objective.uniform_wce * (uniform_wce / MAX_ABS_PRODUCT)
    )
    return Metrics(
        total_cases=SIGNED_TOTAL,
        error_cases=error_cases,
        ER=error_cases/SIGNED_TOTAL,
        MED=med,
        NED=med/MAX_ABS_PRODUCT,
        MRED=uniform_mred,
        MRED_total=uniform_mred_total,
        WCE=uniform_wce,
        RMSE=math.sqrt(uniform_mse),
        bias=float(error.mean()),
        zero_violations=zero_violations,
        symmetry_violations=symmetry_violations,
        objective_score=score,
        workload_ER=workload_er,
        workload_MED=workload_med,
        workload_NED=workload_ned,
        workload_MRED=workload_mred,
        workload_MRED_total=workload_mred_total,
        workload_WCE=workload_wce,
        workload_RMSE=workload_rmse,
        workload_bias=workload_bias,
        workload_zero_violation_probability=zero_v_prob,
        workload_zero_violation_rate=zero_v_rate,
        MSE=uniform_mse,
        NMSE=uniform_nmse,
        workload_MSE=workload_mse,
        workload_NMSE=workload_nmse,
        workload_bias_squared_normalized=normalized_bias_squared,
        workload_conditional_bias_a_rms=math.sqrt(max(conditional_a_squared,0.0)),
        workload_conditional_bias_b_rms=math.sqrt(max(conditional_b_squared,0.0)),
        workload_conditional_bias_excess_normalized=conditional_excess_normalized,
        workload_bias_accumulation_penalty=bias_penalty,
        workload_conditional_bias_penalty=conditional_bias_penalty,
        workload_gemm_NMSE=workload_gemm_nmse,
        workload_predicted_dot_RMSE=predicted_dot_rmse,
        workload_bias_drift_sigma=bias_drift_sigma,
    )

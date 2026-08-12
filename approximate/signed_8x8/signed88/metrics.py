from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .common import LL_STATES, MAX_ABS_PRODUCT, Metrics, ObjectiveWeights, SIGNED_TOTAL
from .data import CalibrationProfile, UNIFORM_NONZERO_PROB, UNIFORM_RED_WEIGHT, UNIFORM_ZERO_PROB


def evaluate_design(design, inits: Mapping[str,str], profile: CalibrationProfile, objective: ObjectiveWeights) -> Metrics:
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
    workload_rmse = math.sqrt(float(np.sum(p * np.square(row_error.astype(np.float64)))))
    workload_wce = int(row_ed.max()) if len(row_ed) else 0
    zero = profile.exact == 0
    zero_v_prob = float(np.sum(p[zero & row_mask]))
    zero_v_rate = zero_v_prob / max(profile.zero_probability,1e-15)
    workload_ned = workload_med / MAX_ABS_PRODUCT

    score = (
        objective.workload_mred * workload_mred
        + objective.workload_er * workload_er
        + objective.workload_ned * workload_ned
        + objective.workload_bias * abs(workload_bias) / MAX_ABS_PRODUCT
        + objective.uniform_mred * uniform_mred
    )
    return Metrics(
        total_cases=SIGNED_TOTAL,
        error_cases=error_cases,
        ER=error_cases/SIGNED_TOTAL,
        MED=med,
        NED=med/MAX_ABS_PRODUCT,
        MRED=uniform_mred,
        MRED_total=uniform_mred_total,
        WCE=int(ed.max()),
        RMSE=math.sqrt(float(np.mean(np.square(error.astype(np.float64))))),
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
    )

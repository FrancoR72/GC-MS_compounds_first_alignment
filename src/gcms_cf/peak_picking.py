from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass(frozen=True)
class Peak1D:
    apex_idx: int
    apex_rt: float
    apex_height: float
    left_idx: int
    right_idx: int
    area: float

def pick_peaks_1d(
    rt: np.ndarray,
    y: np.ndarray,
    *,
    min_rel_height: float = 0.05,
    min_abs_height: float = 0.0,
    min_width_scans: int = 3,
) -> List[Peak1D]:
    """
    Peak picking molto semplice:
    - apex = massimo locale
    - soglia: apex >= max(y)*min_rel_height e >= min_abs_height
    - boundaries: estende a sinistra/destra finché y scende sotto 10% dell'apice
    """
    if rt.size < 3:
        return []

    y = np.asarray(y, dtype=float)
    rt = np.asarray(rt, dtype=float)

    y_max = float(np.max(y)) if y.size else 0.0
    thr = max(min_abs_height, y_max * float(min_rel_height))

    peaks: List[Peak1D] = []

    for i in range(1, len(y) - 1):
        if y[i] < thr:
            continue
        if not (y[i] > y[i-1] and y[i] >= y[i+1]):
            continue

        apex = float(y[i])
        # trova boundary dove scende sotto 10% dell'apice
        cut = apex * 0.10

        l = i
        while l > 0 and y[l] > cut:
            l -= 1

        r = i
        while r < len(y) - 1 and y[r] > cut:
            r += 1

        if (r - l) < min_width_scans:
            continue

        # area trapezi
        area = float(np.trapz(y[l:r+1], rt[l:r+1]))

        peaks.append(Peak1D(
            apex_idx=i,
            apex_rt=float(rt[i]),
            apex_height=apex,
            left_idx=l,
            right_idx=r,
            area=area,
        ))

    # ordina per area decrescente
    peaks.sort(key=lambda p: p.area, reverse=True)
    return peaks

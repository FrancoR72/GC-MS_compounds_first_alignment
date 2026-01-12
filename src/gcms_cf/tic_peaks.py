from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .peak_picking import pick_peaks_1d, Peak1D

@dataclass(frozen=True)
class TICWindow:
    apex_rt: float
    left: float
    right: float
    apex_height: float
    area: float

def pick_tic_windows(
    rt: np.ndarray,
    tic: np.ndarray,
    *,
    top_k: int = 20,
    half_width_min: float = 0.6,
    min_separation_min: float = 0.08,
    min_rel_height: float = 0.03,
    min_width_scans: int = 8,
) -> List[TICWindow]:
    """
    Trova picchi nel TIC e restituisce finestre [apex-half_width, apex+half_width].
    Evita duplicati (picchi adiacenti) usando min_separation_min.
    """
    peaks = pick_peaks_1d(
        rt, tic,
        min_rel_height=min_rel_height,
        min_abs_height=0.0,
        min_width_scans=min_width_scans
    )
    if not peaks:
        return []

    # ordina per height decrescente (o area); qui uso height per stabilità
    peaks = sorted(peaks, key=lambda p: p.apex_height, reverse=True)

    chosen: List[Peak1D] = []
    for p in peaks:
        if any(abs(p.apex_rt - q.apex_rt) < min_separation_min for q in chosen):
            continue
        chosen.append(p)
        if len(chosen) >= top_k:
            break

    out: List[TICWindow] = []
    for p in chosen:
        out.append(TICWindow(
            apex_rt=float(p.apex_rt),
            left=float(p.apex_rt - half_width_min),
            right=float(p.apex_rt + half_width_min),
            apex_height=float(p.apex_height),
            area=float(p.area),
        ))

    # ordina per apex_rt crescente (più comodo)
    out.sort(key=lambda w: w.apex_rt)
    return out

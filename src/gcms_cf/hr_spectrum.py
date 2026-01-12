from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .mzdata_io import iter_mzdata_spectra
from .hr_eic import ppm_window

@dataclass(frozen=True)
class ScanSpectrum:
    spectrum_id: int
    rt_min: float
    mz: np.ndarray
    intensity: np.ndarray

def get_nearest_scan_spectrum(xml_path: str, target_rt: float, *, rt_min: float, rt_max: float) -> ScanSpectrum:
    """
    Scorre in streaming e restituisce lo scan con RT più vicino a target_rt dentro [rt_min, rt_max].
    """
    best = None
    best_dt = None

    for sid, rt, mz, inten in iter_mzdata_spectra(xml_path):
        if not np.isfinite(rt):
            continue
        if rt < rt_min or rt > rt_max:
            continue

        dt = abs(float(rt) - float(target_rt))
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = (sid, float(rt), mz.astype(float), inten.astype(float))

    if best is None:
        raise ValueError("Nessuno scan trovato nella finestra RT indicata.")

    sid, rt, mz, inten = best
    return ScanSpectrum(spectrum_id=int(sid), rt_min=float(rt), mz=mz, intensity=inten)

def extract_matched_spectrum(
    scan: ScanSpectrum,
    target_mz: List[float],
    *,
    ppm_tol: float = 10.0,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """
    Estrae dallo scan i picchi che matchano una lista di m/z (±ppm).
    Ritorna:
      - matched: lista (mz_scan, intensity)
      - matched_sum: somma intensità matchate
      - total_sum: somma intensità totale dello scan (TIC dello scan)
    """
    mz = scan.mz
    it = scan.intensity
    if mz.size == 0:
        return [], 0.0, 0.0

    order = np.argsort(mz)
    mz_s = mz[order]
    it_s = it[order]

    matched = []
    matched_sum = 0.0
    total_sum = float(np.sum(it_s))

    for m0 in target_mz:
        lo, hi = ppm_window(float(m0), float(ppm_tol))
        i1 = int(np.searchsorted(mz_s, lo, side="left"))
        i2 = int(np.searchsorted(mz_s, hi, side="right"))
        if i2 <= i1:
            continue
        # se ci sono più picchi nel range, prendo quello più intenso
        seg = it_s[i1:i2]
        j = int(np.argmax(seg))
        mz_pick = float(mz_s[i1 + j])
        it_pick = float(seg[j])
        matched.append((mz_pick, it_pick))
        matched_sum += it_pick

    # ordina per intensità decrescente
    matched.sort(key=lambda t: t[1], reverse=True)
    return matched, matched_sum, total_sum

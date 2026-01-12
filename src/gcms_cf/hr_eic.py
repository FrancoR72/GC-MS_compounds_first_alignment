from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from .mzdata_io import iter_mzdata_spectra

@dataclass(frozen=True)
class SeedEIC:
    mz: float
    eic: np.ndarray  # intensità vs RT

@dataclass(frozen=True)
class SeedEICResult:
    rt: np.ndarray
    seeds: List[SeedEIC]
    ppm_tol: float

def ppm_window(mz: float, ppm: float) -> Tuple[float, float]:
    delta = mz * ppm * 1e-6
    return (mz - delta, mz + delta)

def _pick_top_seeds(mz: np.ndarray, inten: np.ndarray, *, top_n: int = 80, min_intensity: float = 0.0) -> List[float]:
    if mz.size == 0:
        return []
    mask = inten > float(min_intensity)
    if not np.any(mask):
        return []
    mz2 = mz[mask]
    it2 = inten[mask]
    idx = np.argsort(it2)[::-1][:top_n]
    seeds = [float(mz2[i]) for i in idx]
    # elimina duplicati molto vicini (stessa specie) con tolleranza relativa grezza (1e-5 ~ 10 ppm a 1k)
    seeds_sorted = sorted(seeds)
    out = []
    for m in seeds_sorted:
        if not out:
            out.append(m)
        else:
            if abs(m - out[-1]) > max(1e-5, out[-1] * 3e-6):  # ~3 ppm
                out.append(m)
    return out

def build_seed_eics_from_mzdata(
    xml_path: str,
    *,
    rt_min: float,
    rt_max: float,
    seed_rt: float,
    ppm_tol: float = 10.0,
    top_n_seeds: int = 80,
    min_seed_intensity: float = 0.0,
) -> SeedEICResult:
    """
    HR-friendly:
    1) trova lo scan più vicino a seed_rt (dentro rt_min..rt_max)
    2) prende i top_n_seeds m/z più intensi (seed)
    3) per ciascun seed costruisce EIC sommando intensità entro ±ppm_tol ad ogni scan
    """
    # prima pass: raccogli scans della finestra e trova scan più vicino a seed_rt
    scans = []
    best = None
    best_dt = None

    for sid, rt, mz, inten in iter_mzdata_spectra(xml_path):
        if not np.isfinite(rt):
            continue
        if rt < rt_min or rt > rt_max:
            continue
        scans.append((float(rt), mz.astype(float), inten.astype(float)))
        dt = abs(float(rt) - float(seed_rt))
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = (mz, inten)

    if not scans:
        return SeedEICResult(rt=np.array([], dtype=float), seeds=[], ppm_tol=float(ppm_tol))

    if best is None:
        # fallback: usa il primo scan
        best = (scans[0][1], scans[0][2])

    seed_mz = _pick_top_seeds(best[0], best[1], top_n=top_n_seeds, min_intensity=min_seed_intensity)

    rt_arr = np.array([s[0] for s in scans], dtype=float)

    # pre-alloc EIC arrays
    eics = [np.zeros(len(scans), dtype=float) for _ in seed_mz]

    # per velocità: ordina mz per scan e usa ricerca binaria
    for t_idx, (rt, mz, inten) in enumerate(scans):
        if mz.size == 0:
            continue
        order = np.argsort(mz)
        mz_s = mz[order]
        it_s = inten[order]

        for k, m0 in enumerate(seed_mz):
            lo, hi = ppm_window(m0, ppm_tol)
            i1 = np.searchsorted(mz_s, lo, side="left")
            i2 = np.searchsorted(mz_s, hi, side="right")
            if i2 > i1:
                eics[k][t_idx] = float(np.sum(it_s[i1:i2]))

    seeds = [SeedEIC(mz=m, eic=eics[i]) for i, m in enumerate(seed_mz)]
    return SeedEICResult(rt=rt_arr, seeds=seeds, ppm_tol=float(ppm_tol))

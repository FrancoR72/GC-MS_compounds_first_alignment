from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np

from .mzdata_io import iter_mzdata_spectra

@dataclass(frozen=True)
class EICResult:
    rt: np.ndarray                  # shape (n_scans,)
    eic_by_bin: Dict[float, np.ndarray]  # bin_center -> intensities (n_scans,)
    bin_width: float

def _bin_center(mz: float, bin_width: float) -> float:
    return round((mz / bin_width)) * bin_width

def build_binned_eic_from_mzdata(
    xml_path: str,
    *,
    rt_min: float,
    rt_max: float,
    bin_width: float = 0.1,
    max_bins: Optional[int] = None,
) -> EICResult:
    """
    Costruisce EIC per bin m/z in una finestra RT.
    Ogni bin raccoglie la somma delle intensità dei picchi che cadono in quel bin.
    """
    rts: List[float] = []
    eic_by_bin: Dict[float, List[float]] = {}
    seen_bins: Dict[float, None] = {}

    # streaming spectra
    for sid, rt, mz, inten in iter_mzdata_spectra(xml_path):
        if not np.isfinite(rt):
            continue
        if rt < rt_min or rt > rt_max:
            continue

        rts.append(float(rt))

        # inizializza tutti i bin visti finora con 0 per questo scan
        for b in seen_bins.keys():
            eic_by_bin[b].append(0.0)

        if mz.size:
            # somma intensità per bin (per questo scan)
            accum: Dict[float, float] = {}
            for mzi, inti in zip(mz, inten):
                b = _bin_center(float(mzi), bin_width)
                accum[b] = accum.get(b, 0.0) + float(inti)

            # aggiungi i bin nuovi
            for b in accum.keys():
                if b not in seen_bins:
                    # nuovo bin: backfill zeri per scans precedenti
                    seen_bins[b] = None
                    eic_by_bin[b] = [0.0] * (len(rts) - 1)  # per scans già letti
                    eic_by_bin[b].append(accum[b])
                else:
                    # bin già esistente: sovrascrivi l’ultimo 0 con il valore
                    eic_by_bin[b][-1] = accum[b]

        # limita numero di bin (opzionale)
        if max_bins is not None and len(seen_bins) > max_bins:
            # tieni solo i bin con maggiore area accumulata finora
            areas = [(b, sum(eic_by_bin[b])) for b in list(seen_bins.keys())]
            areas.sort(key=lambda t: t[1], reverse=True)
            keep = set(b for b, _ in areas[:max_bins])
            drop = [b for b in list(seen_bins.keys()) if b not in keep]
            for b in drop:
                seen_bins.pop(b, None)
                eic_by_bin.pop(b, None)

    rt_arr = np.array(rts, dtype=float)
    # convert lists -> arrays
    eic_arr_by_bin = {b: np.array(v, dtype=float) for b, v in eic_by_bin.items()}

    return EICResult(rt=rt_arr, eic_by_bin=eic_arr_by_bin, bin_width=bin_width)

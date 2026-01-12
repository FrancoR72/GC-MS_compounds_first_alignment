from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from .models import Peak, Compound
from .mzdata_io import compute_tic_mzdata, iter_mzdata_spectra
from .tic_peaks import pick_tic_windows, TICWindow
from .peak_picking import pick_peaks_1d

@dataclass(frozen=True)
class CompoundCandidateHR:
    sample_id: str
    rt_apex: float
    window_left: float
    window_right: float
    n_ions: int
    area_cluster: float
    cluster_purity: float
    scan_id: int
    scan_rt: float
    scan_purity: float
    matched_peaks: List[Tuple[float, float]]  # (mz, intensity), ordinati per intensità

def ppm_window(mz: float, ppm: float) -> Tuple[float, float]:
    d = mz * ppm * 1e-6
    return mz - d, mz + d

def _pick_top_seeds(mz: np.ndarray, inten: np.ndarray, *, top_n: int = 120) -> List[float]:
    if mz.size == 0:
        return []
    idx = np.argsort(inten)[::-1][:top_n]
    seeds = [float(mz[i]) for i in idx]
    seeds.sort()
    # dedup molto vicino (~3 ppm) per evitare duplicati
    out: List[float] = []
    for m in seeds:
        if not out:
            out.append(m)
        else:
            if abs(m - out[-1]) > max(1e-6, out[-1] * 3e-6):
                out.append(m)
    return out

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)
    if x.size < 3 or y.size < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x*x).sum()) * np.sqrt((y*y).sum())
    if den == 0:
        return 0.0
    return float((x*y).sum() / den)

def _connected_components(adj: Dict[int, List[int]], n: int) -> List[List[int]]:
    seen = [False]*n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, []):
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps

def _build_seed_eics_from_scans(
    scans: List[Tuple[float, np.ndarray, np.ndarray]],
    seed_rt: float,
    *,
    ppm_tol: float,
    top_n_seeds: int,
) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
    # trova scan più vicino
    rts = np.array([s[0] for s in scans], dtype=float)
    j = int(np.argmin(np.abs(rts - seed_rt)))
    mz0, it0 = scans[j][1], scans[j][2]

    seeds = _pick_top_seeds(mz0, it0, top_n=top_n_seeds)
    eics = [np.zeros(len(scans), dtype=float) for _ in seeds]

    # costruisci EIC
    for t_idx, (rt, mz, inten) in enumerate(scans):
        if mz.size == 0:
            continue
        order = np.argsort(mz)
        mz_s = mz[order]
        it_s = inten[order]
        for k, m0 in enumerate(seeds):
            lo, hi = ppm_window(m0, ppm_tol)
            i1 = np.searchsorted(mz_s, lo, side="left")
            i2 = np.searchsorted(mz_s, hi, side="right")
            if i2 > i1:
                eics[k][t_idx] = float(np.sum(it_s[i1:i2]))

    return rts, seeds, eics

def _extract_scan_level_matched_spectrum(
    scans: List[Tuple[float, np.ndarray, np.ndarray]],
    target_rt: float,
    target_mz: List[float],
    *,
    ppm_tol: float,
) -> Tuple[int, float, List[Tuple[float, float]], float]:
    """
    Usa lo scan più vicino a target_rt nella finestra.
    Ritorna: (scan_idx, scan_rt, matched_peaks, scan_purity)
    """
    rts = np.array([s[0] for s in scans], dtype=float)
    scan_idx = int(np.argmin(np.abs(rts - target_rt)))
    scan_rt, mz, inten = scans[scan_idx]
    if mz.size == 0:
        return scan_idx, float(scan_rt), [], 0.0

    order = np.argsort(mz)
    mz_s = mz[order]
    it_s = inten[order]
    total = float(np.sum(it_s)) if it_s.size else 0.0

    matched = []
    matched_sum = 0.0
    for m0 in target_mz:
        lo, hi = ppm_window(m0, ppm_tol)
        i1 = int(np.searchsorted(mz_s, lo, side="left"))
        i2 = int(np.searchsorted(mz_s, hi, side="right"))
        if i2 <= i1:
            continue
        seg = it_s[i1:i2]
        j = int(np.argmax(seg))
        matched.append((float(mz_s[i1+j]), float(seg[j])))
        matched_sum += float(seg[j])

    matched.sort(key=lambda t: t[1], reverse=True)
    purity = (matched_sum / total) if total > 0 else 0.0
    return scan_idx, float(scan_rt), matched, float(purity)

def extract_compounds_from_mzdata(
    xml_path: str,
    *,
    sample_id: str,
    top_k_windows: int = 20,
    half_width_min: float = 0.6,
    ppm_tol: float = 10.0,
    top_n_seeds: int = 120,
    hit_rt_tol: float = 0.05,
    hit_min_rel_height: float = 0.08,
    max_hits: int = 60,
    min_corr: float = 0.85,
) -> Tuple[List[Compound], "object"]:
    """
    Estrae una lista di Compound (baseline HR) da un singolo mzData.xml.
    Ritorna anche una tabella di riepilogo (DataFrame se pandas disponibile).
    """
    # 1) TIC (1 pass)
    rt, tic, n_peaks = compute_tic_mzdata(xml_path)
    windows = pick_tic_windows(
        rt, tic,
        top_k=top_k_windows,
        half_width_min=half_width_min,
        min_separation_min=0.08,
        min_rel_height=0.03,
        min_width_scans=8,
    )

    # 2) raccogli scans per finestra (1 pass sul file)
    # windows non sovrapposte troppo: check semplice per ogni scan
    win_scans: List[List[Tuple[float, np.ndarray, np.ndarray]]] = [[] for _ in windows]

    for sid, rti, mz, inten in iter_mzdata_spectra(xml_path):
        if not np.isfinite(rti):
            continue
        rtf = float(rti)
        for w_idx, w in enumerate(windows):
            if w.left <= rtf <= w.right:
                win_scans[w_idx].append((rtf, mz.astype(float), inten.astype(float)))

    compounds: List[Compound] = []
    rows = []

    # 3) per ogni finestra: seed EIC -> hits -> clustering -> scan spectrum
    for w_idx, w in enumerate(windows):
        scans = win_scans[w_idx]
        if len(scans) < 10:
            continue
        scans.sort(key=lambda t: t[0])

        rts, seeds, eics = _build_seed_eics_from_scans(scans, w.apex_rt, ppm_tol=ppm_tol, top_n_seeds=top_n_seeds)

        # hits: seed con picco vicino a apex
        hits = []
        for m0, eic in zip(seeds, eics):
            peaks = pick_peaks_1d(rts, eic, min_rel_height=hit_min_rel_height, min_width_scans=4)
            if not peaks:
                continue
            best = min(peaks, key=lambda p: abs(p.apex_rt - w.apex_rt))
            if abs(best.apex_rt - w.apex_rt) <= hit_rt_tol:
                hits.append((m0, float(best.area), float(best.apex_rt), eic))

        if not hits:
            continue

        hits.sort(key=lambda t: t[1], reverse=True)
        hits = hits[:max_hits]

        # clustering per correlazione
        X = [np.log1p(h[3]) for h in hits]
        n = len(X)
        adj: Dict[int, List[int]] = {}
        for i in range(n):
            for j in range(i+1, n):
                c = _pearson(X[i], X[j])
                if c >= min_corr:
                    adj.setdefault(i, []).append(j)
                    adj.setdefault(j, []).append(i)
        comps = _connected_components(adj, n)

        def comp_area(comp):
            return sum(hits[i][1] for i in comp)

        comps.sort(key=comp_area, reverse=True)
        best_comp = comps[0]
        area_cluster = comp_area(best_comp)
        area_total = sum(h[1] for h in hits)
        cluster_purity = (area_cluster / area_total) if area_total > 0 else 0.0

        cluster_mz = [hits[i][0] for i in best_comp]
        cluster_mz.sort()

        scan_idx, scan_rt, matched, scan_purity = _extract_scan_level_matched_spectrum(
            scans, w.apex_rt, cluster_mz, ppm_tol=ppm_tol
        )

        # crea Compound (spectrum = matched peaks)
        spectrum_peaks = [Peak(mz=float(m), intensity=float(ii)) for m, ii in matched]
        compound_id = f"{sample_id}_RT{w.apex_rt:.3f}".replace(".", "p")

        comp = Compound(
            compound_id=compound_id,
            sample_id=sample_id,
            rt=float(w.apex_rt),
            ri=None,
            area=float(area_cluster),
            purity=float(scan_purity),  # qui usiamo la purity scan-level (più conservativa)
            spectrum=spectrum_peaks,
        )
        compounds.append(comp)

        rows.append({
            "compound_id": compound_id,
            "rt_apex": float(w.apex_rt),
            "n_ions_cluster": int(len(best_comp)),
            "area_cluster": float(area_cluster),
            "cluster_purity": float(cluster_purity),
            "scan_rt": float(scan_rt),
            "scan_purity": float(scan_purity),
            "matched_peaks": int(len(matched)),
        })

    # summary table
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows).sort_values(["rt_apex"]).reset_index(drop=True)
        return compounds, df
    except Exception:
        return compounds, rows

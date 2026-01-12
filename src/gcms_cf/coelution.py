from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class IonPeak:
    mz_bin: float
    apex_rt: float
    area: float
    apex_height: float
    left_idx: int
    right_idx: int

@dataclass(frozen=True)
class CompoundCandidate:
    target_rt: float
    rt_left: float
    rt_right: float
    ions: List[IonPeak]
    spectrum: List[Tuple[float, float]]  # (mz_bin, intensity_at_apex)
    area_sum: float
    purity: float  # 0..1

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return 0.0
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
    den = (np.sqrt((x*x).sum()) * np.sqrt((y*y).sum()))
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

def cluster_coeluting_ions(
    rt: np.ndarray,
    eic_by_bin: Dict[float, np.ndarray],
    ion_peaks: List[IonPeak],
    *,
    min_corr: float = 0.85,
    log1p: bool = True,
) -> List[List[IonPeak]]:
    """
    Clustering per co-eluzione:
    - per ogni coppia di bin si calcola la correlazione Pearson sul tratto [left:right] del riferimento
    - si crea un grafo con edge se corr>=min_corr
    - componenti connesse = cluster di ioni
    """
    if not ion_peaks:
        return []

    # finestra RT comune = quella del picco di riferimento (il primo della lista)
    ref = ion_peaks[0]
    l, r = ref.left_idx, ref.right_idx

    bins = [p.mz_bin for p in ion_peaks]
    segs = []
    for b in bins:
        y = eic_by_bin[b][l:r+1]
        if log1p:
            y = np.log1p(y)
        segs.append(y)

    n = len(segs)
    adj: Dict[int, List[int]] = {}

    for i in range(n):
        for j in range(i+1, n):
            c = _pearson(segs[i], segs[j])
            if c >= min_corr:
                adj.setdefault(i, []).append(j)
                adj.setdefault(j, []).append(i)

    comps_idx = _connected_components(adj, n)
    clusters: List[List[IonPeak]] = []
    for comp in comps_idx:
        clusters.append([ion_peaks[k] for k in comp])

    # ordina cluster per area totale decrescente
    clusters.sort(key=lambda ions: sum(p.area for p in ions), reverse=True)
    return clusters

def build_compound_candidate(
    rt: np.ndarray,
    eic_by_bin: Dict[float, np.ndarray],
    cluster: List[IonPeak],
    *,
    target_rt: float,
    purity_den_ions: List[IonPeak],
) -> CompoundCandidate:
    """
    Costruisce un candidato compound:
    - spectrum = intensità dei bin al tempo più vicino a target_rt
    - area_sum = somma aree dei bin nel cluster
    - purity = area_sum / somma aree dei bin considerati (purity_den_ions)
    """
    if not cluster:
        raise ValueError("Cluster vuoto")

    # uso finestra del primo ion peak
    l, r = cluster[0].left_idx, cluster[0].right_idx
    rt_left = float(rt[l])
    rt_right = float(rt[r])

    # indice scan più vicino a target_rt
    apex_idx = int(np.argmin(np.abs(rt - target_rt)))

    spectrum = []
    for p in sorted(cluster, key=lambda x: x.area, reverse=True):
        inten = float(eic_by_bin[p.mz_bin][apex_idx])
        spectrum.append((p.mz_bin, inten))

    area_sum = float(sum(p.area for p in cluster))
    den = float(sum(p.area for p in purity_den_ions)) if purity_den_ions else area_sum
    purity = float(area_sum / den) if den > 0 else 0.0

    return CompoundCandidate(
        target_rt=float(target_rt),
        rt_left=rt_left,
        rt_right=rt_right,
        ions=sorted(cluster, key=lambda x: x.area, reverse=True),
        spectrum=spectrum,
        area_sum=area_sum,
        purity=purity,
    )

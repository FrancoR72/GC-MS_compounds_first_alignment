from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from .models import Compound, Peak

@dataclass
class Feature:
    feature_id: str
    rt_ref: float  # RT di riferimento (minuti)
    by_sample_area: Dict[str, float]  # sample_id -> area (0 se assente)

def _cosine_similarity(peaks_a: List[Peak], peaks_b: List[Peak], *, mz_tol: float = 0.1) -> float:
    """
    Cosine similarity tra due spettri centroid (liste di Peak).
    Matching su m/z con tolleranza mz_tol (unità m/z).
    """
    if not peaks_a or not peaks_b:
        return 0.0

    a = sorted(peaks_a, key=lambda p: p.mz)
    b = sorted(peaks_b, key=lambda p: p.mz)

    i = j = 0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    # norme complete
    for p in a:
        norm_a += p.intensity * p.intensity
    for p in b:
        norm_b += p.intensity * p.intensity

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # dot product con matching tolleranza
    while i < len(a) and j < len(b):
        mz_a = a[i].mz
        mz_b = b[j].mz
        diff = mz_a - mz_b

        if abs(diff) <= mz_tol:
            dot += a[i].intensity * b[j].intensity
            i += 1
            j += 1
        elif diff < -mz_tol:
            i += 1
        else:
            j += 1

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

def _dlog10_area(a1: float, a2: float, eps: float = 1e-12) -> float:
    return abs(math.log10(max(a1, eps)) - math.log10(max(a2, eps)))

def _assign_to_cluster(
    c: Compound,
    clusters: List[Dict],
    *,
    rt_tol: float,
    min_cosine: float,
    mz_tol: float,
    max_dlog10_area: float,
) -> Optional[int]:
    """
    Trova cluster compatibile con:
    - |rt - rt_ref| <= rt_tol
    - cosine(rep_spectrum, c_spectrum) >= min_cosine
    - |log10(area) - log10(rep_area)| <= max_dlog10_area

    Sceglie il cluster con distanza RT minima.
    """
    best_idx = None
    best_dist = None

    for i, cl in enumerate(clusters):
        dist = abs(c.rt - cl["rt_ref"])
        if dist > rt_tol:
            continue

        rep: Compound = cl["rep"]
        cos = _cosine_similarity(rep.spectrum, c.spectrum, mz_tol=mz_tol)
        if cos < min_cosine:
            continue

        dlog = _dlog10_area(rep.area, c.area)
        if dlog > max_dlog10_area:
            continue

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx

def align_compounds_rt_only(
    compounds: List[Compound],
    *,
    rt_tol: float = 1.0,           # default: 1 minuto (come richiesto)
    feature_prefix: str = "F",
    area_agg: str = "max",         # "max" oppure "sum"
    # nuovi filtri "reali"
    min_cosine: float = 0.80,
    mz_tol: float = 0.1,
    max_dlog10_area: float = 1.0,
) -> List[Feature]:
    """
    Allineamento baseline:
    - RT tolerance (rt_tol)
    - filtro cosine similarity (min_cosine, mz_tol)
    - filtro ordine di grandezza aree (max_dlog10_area)
    """
    if area_agg not in ("max", "sum"):
        raise ValueError("area_agg deve essere 'max' oppure 'sum'")

    comps_sorted = sorted(compounds, key=lambda x: x.rt)

    clusters: List[Dict] = []
    for c in comps_sorted:
        idx = _assign_to_cluster(
            c, clusters,
            rt_tol=rt_tol,
            min_cosine=min_cosine,
            mz_tol=mz_tol,
            max_dlog10_area=max_dlog10_area,
        )
        if idx is None:
            clusters.append({"rt_ref": c.rt, "members": [c], "rep": c})
        else:
            clusters[idx]["members"].append(c)
            # aggiorno rt_ref come media (semplice)
            rts = [m.rt for m in clusters[idx]["members"]]
            clusters[idx]["rt_ref"] = sum(rts) / len(rts)
            # rep: tengo quello con area maggiore (semplice “representative”)
            rep = clusters[idx]["rep"]
            if c.area > rep.area:
                clusters[idx]["rep"] = c

    features: List[Feature] = []
    for k, cl in enumerate(clusters, start=1):
        feature_id = f"{feature_prefix}{k:06d}"
        rt_ref = cl["rt_ref"]

        by_sample: Dict[str, float] = {}
        for m in cl["members"]:
            sid = m.sample_id
            if sid not in by_sample:
                by_sample[sid] = m.area
            else:
                if area_agg == "max":
                    by_sample[sid] = max(by_sample[sid], m.area)
                else:
                    by_sample[sid] = by_sample[sid] + m.area

        features.append(Feature(feature_id=feature_id, rt_ref=rt_ref, by_sample_area=by_sample))

    return features

def features_to_table(features: List[Feature], *, fill_missing: float = 0.0):
    sample_ids = sorted({sid for f in features for sid in f.by_sample_area.keys()})
    header = ["feature_id", "rt_ref"] + sample_ids
    rows = []
    for f in features:
        row = [f.feature_id, f.rt_ref]
        for sid in sample_ids:
            row.append(f.by_sample_area.get(sid, fill_missing))
        rows.append(row)

    try:
        import pandas as pd  # type: ignore
        return pd.DataFrame(rows, columns=header)
    except Exception:
        return header, rows

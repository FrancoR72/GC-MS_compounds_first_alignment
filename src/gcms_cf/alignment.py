from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from .models import Compound, Peak

@dataclass
class Feature:
    feature_id: str
    rt_ref: float
    ri_ref: Optional[float]
    by_sample_area: Dict[str, float]

def _cosine_similarity(peaks_a: List[Peak], peaks_b: List[Peak], *, mz_tol: float = 0.1) -> float:
    if not peaks_a or not peaks_b:
        return 0.0

    a = sorted(peaks_a, key=lambda p: p.mz)
    b = sorted(peaks_b, key=lambda p: p.mz)

    dot = 0.0
    norm_a = sum(p.intensity * p.intensity for p in a)
    norm_b = sum(p.intensity * p.intensity for p in b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    i = j = 0
    while i < len(a) and j < len(b):
        diff = a[i].mz - b[j].mz
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
    use_ri: bool,
    ri_tol: float,
    min_cosine: float,
    mz_tol: float,
    max_dlog10_area: float,
) -> Optional[int]:
    best_idx = None
    best_dist = None

    for i, cl in enumerate(clusters):
        dist = abs(c.rt - cl["rt_ref"])
        if dist > rt_tol:
            continue

        rep: Compound = cl["rep"]

        # RI opzionale
        if use_ri:
            if c.ri is None or rep.ri is None:
                continue
            if abs(c.ri - rep.ri) > ri_tol:
                continue

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
    rt_tol: float = 1.0,            # minuti
    use_ri: bool = False,
    ri_tol: float = 20.0,
    feature_prefix: str = "F",
    area_agg: str = "max",          # "max" oppure "sum"
    min_cosine: float = 0.80,
    mz_tol: float = 0.1,
    max_dlog10_area: float = 1.0,
) -> List[Feature]:
    if area_agg not in ("max", "sum"):
        raise ValueError("area_agg deve essere 'max' oppure 'sum'")

    comps_sorted = sorted(compounds, key=lambda x: x.rt)

    clusters: List[Dict] = []
    for c in comps_sorted:
        idx = _assign_to_cluster(
            c, clusters,
            rt_tol=rt_tol,
            use_ri=use_ri,
            ri_tol=ri_tol,
            min_cosine=min_cosine,
            mz_tol=mz_tol,
            max_dlog10_area=max_dlog10_area,
        )
        if idx is None:
            clusters.append({"rt_ref": c.rt, "members": [c], "rep": c})
        else:
            clusters[idx]["members"].append(c)

            # aggiorno rt_ref come media
            rts = [m.rt for m in clusters[idx]["members"]]
            clusters[idx]["rt_ref"] = sum(rts) / len(rts)

            # representative = area maggiore
            rep = clusters[idx]["rep"]
            if c.area > rep.area:
                clusters[idx]["rep"] = c

    features: List[Feature] = []
    for k, cl in enumerate(clusters, start=1):
        feature_id = f"{feature_prefix}{k:06d}"
        rt_ref = cl["rt_ref"]

        ris = [m.ri for m in cl["members"] if m.ri is not None]
        ri_ref = (sum(ris) / len(ris)) if ris else None

        by_sample: Dict[str, float] = {}
        for m in cl["members"]:
            sid = m.sample_id
            if sid not in by_sample:
                by_sample[sid] = m.area
            else:
                by_sample[sid] = max(by_sample[sid], m.area) if area_agg == "max" else (by_sample[sid] + m.area)

        features.append(Feature(feature_id=feature_id, rt_ref=rt_ref, ri_ref=ri_ref, by_sample_area=by_sample))

    return features

def features_to_table(features: List[Feature], *, fill_missing: float = 0.0):
    sample_ids = sorted({sid for f in features for sid in f.by_sample_area.keys()})
    header = ["feature_id", "rt_ref", "ri_ref"] + sample_ids

    rows = []
    for f in features:
        row = [f.feature_id, f.rt_ref, f.ri_ref]
        for sid in sample_ids:
            row.append(f.by_sample_area.get(sid, fill_missing))
        rows.append(row)

    try:
        import pandas as pd  # type: ignore
        return pd.DataFrame(rows, columns=header)
    except Exception:
        return header, rows

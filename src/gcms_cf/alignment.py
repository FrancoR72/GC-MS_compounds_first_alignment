from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from .models import Compound, Peak

@dataclass
class Feature:
    feature_id: str
    rt_ref: float
    ri_ref: Optional[float]
    by_sample_area: Dict[str, float]

@dataclass
class Cluster:
    feature_id: str
    members: List[Compound]
    rt_ref: float
    ri_ref: Optional[float]
    rep: Compound
    by_sample_area: Dict[str, float]

# -----------------------
# helpers
# -----------------------
def _mz_tol_ppm(mz: float, ppm: float) -> float:
    return float(mz) * float(ppm) * 1e-6

def _cosine_similarity(
    peaks_a: List[Peak],
    peaks_b: List[Peak],
    *,
    mz_tol: float = 0.01,
    mz_ppm: Optional[float] = None
) -> float:
    """
    Cosine similarity tra due spettri centroid.
    Matching su m/z:
      - se mz_ppm è impostato: tol = ±(mz_ppm ppm)
      - altrimenti usa mz_tol assoluto (Da)
    """
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
        mz_a = float(a[i].mz)
        mz_b = float(b[j].mz)
        diff = mz_a - mz_b

        if mz_ppm is not None:
            tol = _mz_tol_ppm(max(mz_a, mz_b), mz_ppm)
        else:
            tol = float(mz_tol)

        if abs(diff) <= tol:
            dot += float(a[i].intensity) * float(b[j].intensity)
            i += 1
            j += 1
        elif diff < -tol:
            i += 1
        else:
            j += 1

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

def _dlog10_area(a1: float, a2: float, eps: float = 1e-12) -> float:
    return abs(math.log10(max(a1, eps)) - math.log10(max(a2, eps)))

def _safe_div(x: float, denom: float, default: float = 0.0) -> float:
    return x / denom if denom and denom > 0 else default

def _rep_quality(c: Compound) -> float:
    return max(c.area, 0.0) * max(c.purity, 0.0)

def _build_by_sample_area(members: List[Compound], *, area_agg: str) -> Dict[str, float]:
    by_sample: Dict[str, float] = {}
    for m in members:
        sid = m.sample_id
        if sid not in by_sample:
            by_sample[sid] = m.area
        else:
            by_sample[sid] = max(by_sample[sid], m.area) if area_agg == "max" else (by_sample[sid] + m.area)
    return by_sample

def _update_refs_weighted(cluster: Cluster):
    members = cluster.members
    w_sum = sum(max(m.area, 0.0) for m in members)
    if w_sum <= 0:
        cluster.rt_ref = sum(m.rt for m in members) / len(members)
        ris = [m.ri for m in members if m.ri is not None]
        cluster.ri_ref = (sum(ris) / len(ris)) if ris else None
        return

    cluster.rt_ref = sum(m.rt * max(m.area, 0.0) for m in members) / w_sum

    ris = [(m.ri, m.area) for m in members if m.ri is not None]
    if ris:
        ri_w_sum = sum(max(w, 0.0) for _, w in ris)
        if ri_w_sum > 0:
            cluster.ri_ref = sum(ri * max(w, 0.0) for ri, w in ris) / ri_w_sum
        else:
            cluster.ri_ref = sum(ri for ri, _ in ris) / len(ris)
    else:
        cluster.ri_ref = None

def update_cluster_after_adding(cluster: Cluster, *, area_agg: str = "max"):
    _update_refs_weighted(cluster)
    rep = cluster.rep
    for m in cluster.members:
        if _rep_quality(m) > _rep_quality(rep):
            rep = m
    cluster.rep = rep
    cluster.by_sample_area = _build_by_sample_area(cluster.members, area_agg=area_agg)

def _score_to_cluster(
    c: Compound,
    cl: Cluster,
    *,
    rt_tol: float,
    use_ri: bool,
    ri_tol: float,
    min_cosine: float,
    mz_tol: float,
    mz_ppm: Optional[float],
    max_dlog10_area: float,
    w_rt: float,
    w_ri: float,
    w_cos: float,
    w_dlog: float,
) -> Optional[float]:
    dist_rt = abs(c.rt - cl.rt_ref)
    if dist_rt > rt_tol:
        return None

    if use_ri:
        if c.ri is None or cl.ri_ref is None:
            return None
        dist_ri = abs(c.ri - cl.ri_ref)
        if dist_ri > ri_tol:
            return None
    else:
        dist_ri = 0.0

    cos = _cosine_similarity(cl.rep.spectrum, c.spectrum, mz_tol=mz_tol, mz_ppm=mz_ppm)
    if cos < min_cosine:
        return None

    dlog = _dlog10_area(cl.rep.area, c.area)
    if dlog > max_dlog10_area:
        return None

    rt_term = _safe_div(dist_rt, rt_tol)
    ri_term = _safe_div(dist_ri, ri_tol) if use_ri else 0.0
    cos_term = _safe_div((1.0 - cos), (1.0 - min_cosine), default=1.0)
    dlog_term = _safe_div(dlog, max_dlog10_area)

    return (w_rt * rt_term) + (w_ri * ri_term) + (w_cos * cos_term) + (w_dlog * dlog_term)

# -----------------------
# public API
# -----------------------
def align_compounds_clusters(
    compounds: List[Compound],
    *,
    rt_tol: float = 0.25,
    use_ri: bool = False,
    ri_tol: float = 20.0,
    feature_prefix: str = "F",
    area_agg: str = "max",
    min_cosine: float = 0.78,
    mz_tol: float = 0.01,
    mz_ppm: Optional[float] = None,
    max_dlog10_area: float = 1.0,
    w_rt: float = 1.0,
    w_ri: float = 1.0,
    w_cos: float = 1.0,
    w_dlog: float = 1.0,
) -> List[Cluster]:
    if area_agg not in ("max", "sum"):
        raise ValueError("area_agg deve essere 'max' oppure 'sum'")

    comps_sorted = sorted(compounds, key=lambda x: x.rt)
    clusters: List[Cluster] = []

    for c in comps_sorted:
        best_idx = None
        best_score = None

        for i, cl in enumerate(clusters):
            s = _score_to_cluster(
                c, cl,
                rt_tol=rt_tol, use_ri=use_ri, ri_tol=ri_tol,
                min_cosine=min_cosine,
                mz_tol=mz_tol, mz_ppm=mz_ppm,
                max_dlog10_area=max_dlog10_area,
                w_rt=w_rt, w_ri=w_ri, w_cos=w_cos, w_dlog=w_dlog,
            )
            if s is None:
                continue
            if best_score is None or s < best_score:
                best_score = s
                best_idx = i

        if best_idx is None:
            clusters.append(Cluster(
                feature_id="",
                members=[c],
                rt_ref=c.rt,
                ri_ref=c.ri,
                rep=c,
                by_sample_area={c.sample_id: c.area},
            ))
        else:
            cl = clusters[best_idx]
            cl.members.append(c)
            update_cluster_after_adding(cl, area_agg=area_agg)

    for k, cl in enumerate(clusters, start=1):
        cl.feature_id = f"{feature_prefix}{k:06d}"
        cl.by_sample_area = _build_by_sample_area(cl.members, area_agg=area_agg)

    return clusters

def clusters_to_table(clusters: List[Cluster], *, fill_missing: float = 0.0):
    sample_ids = sorted({sid for cl in clusters for sid in cl.by_sample_area.keys()})
    header = ["feature_id", "rt_ref", "ri_ref"] + sample_ids

    rows = []
    for cl in clusters:
        row = [cl.feature_id, cl.rt_ref, cl.ri_ref]
        for sid in sample_ids:
            row.append(cl.by_sample_area.get(sid, fill_missing))
        rows.append(row)

    try:
        import pandas as pd  # type: ignore
        return pd.DataFrame(rows, columns=header)
    except Exception:
        return header, rows

def rank_candidates_for_feature(
    sample_compounds: List[Compound],
    *,
    rt_ref: float,
    ri_ref: Optional[float],
    rep: Compound,
    rt_tol: float,
    use_ri: bool,
    ri_tol: float,
    min_cosine: float,
    mz_tol: float,
    mz_ppm: Optional[float],
    max_dlog10_area: float,
    w_rt: float = 1.0,
    w_ri: float = 1.0,
    w_cos: float = 1.0,
    w_dlog: float = 1.0,
) -> List[Tuple[Compound, float]]:
    dummy = Cluster(
        feature_id="",
        members=[],
        rt_ref=rt_ref,
        ri_ref=ri_ref,
        rep=rep,
        by_sample_area={},
    )

    scored: List[Tuple[Compound, float]] = []
    for c in sample_compounds:
        s = _score_to_cluster(
            c, dummy,
            rt_tol=rt_tol, use_ri=use_ri, ri_tol=ri_tol,
            min_cosine=min_cosine,
            mz_tol=mz_tol, mz_ppm=mz_ppm,
            max_dlog10_area=max_dlog10_area,
            w_rt=w_rt, w_ri=w_ri, w_cos=w_cos, w_dlog=w_dlog,
        )
        if s is not None:
            scored.append((c, s))

    scored.sort(key=lambda t: t[1])
    return scored

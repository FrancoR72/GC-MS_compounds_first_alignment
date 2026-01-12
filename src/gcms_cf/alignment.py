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

def _safe_div(x: float, denom: float, default: float = 0.0) -> float:
    return x / denom if denom and denom > 0 else default

def _update_refs_weighted(cluster: Dict):
    """Aggiorna rt_ref e ri_ref come medie pesate per area."""
    members: List[Compound] = cluster["members"]
    w_sum = sum(max(m.area, 0.0) for m in members)
    if w_sum <= 0:
        # fallback: media semplice
        cluster["rt_ref"] = sum(m.rt for m in members) / len(members)
        ris = [m.ri for m in members if m.ri is not None]
        cluster["ri_ref"] = (sum(ris) / len(ris)) if ris else None
        return

    cluster["rt_ref"] = sum(m.rt * max(m.area, 0.0) for m in members) / w_sum

    ris = [(m.ri, m.area) for m in members if m.ri is not None]
    if ris:
        ri_w_sum = sum(max(w, 0.0) for _, w in ris)
        if ri_w_sum > 0:
            cluster["ri_ref"] = sum(ri * max(w, 0.0) for ri, w in ris) / ri_w_sum
        else:
            cluster["ri_ref"] = sum(ri for ri, _ in ris) / len(ris)
    else:
        cluster["ri_ref"] = None

def _rep_quality(c: Compound) -> float:
    # se purity in futuro diventa 0..1, questa scelta diventa utile.
    return max(c.area, 0.0) * max(c.purity, 0.0)

def _score_candidate(
    c: Compound,
    cluster: Dict,
    *,
    rt_tol: float,
    use_ri: bool,
    ri_tol: float,
    min_cosine: float,
    mz_tol: float,
    max_dlog10_area: float,
    w_rt: float,
    w_ri: float,
    w_cos: float,
    w_dlog: float,
) -> Optional[float]:
    """
    Ritorna uno score (più basso = meglio) oppure None se non supera i filtri.
    Filtri “hard”:
      - RT entro rt_tol
      - (se use_ri) RI entro ri_tol e non None
      - cosine >= min_cosine
      - dlog10_area <= max_dlog10_area
    """
    rt_ref = cluster["rt_ref"]
    dist_rt = abs(c.rt - rt_ref)
    if dist_rt > rt_tol:
        return None

    ri_ref = cluster.get("ri_ref", None)
    if use_ri:
        if c.ri is None or ri_ref is None:
            return None
        dist_ri = abs(c.ri - ri_ref)
        if dist_ri > ri_tol:
            return None
    else:
        dist_ri = 0.0

    rep: Compound = cluster["rep"]
    cos = _cosine_similarity(rep.spectrum, c.spectrum, mz_tol=mz_tol)
    if cos < min_cosine:
        return None

    dlog = _dlog10_area(rep.area, c.area)
    if dlog > max_dlog10_area:
        return None

    # Score normalizzato (0..~1 per ciascun termine) e pesato
    rt_term = _safe_div(dist_rt, rt_tol)
    ri_term = _safe_div(dist_ri, ri_tol) if use_ri else 0.0

    # più cos è alto meglio; normalizzo in 0..1 rispetto a min_cosine
    # cos_term=0 quando cos=1; cos_term=1 quando cos=min_cosine
    cos_term = _safe_div((1.0 - cos), (1.0 - min_cosine), default=1.0)

    dlog_term = _safe_div(dlog, max_dlog10_area)

    score = (w_rt * rt_term) + (w_ri * ri_term) + (w_cos * cos_term) + (w_dlog * dlog_term)
    return score

def align_compounds_rt_only(
    compounds: List[Compound],
    *,
    rt_tol: float = 1.0,           # minuti (default richiesto)
    use_ri: bool = False,
    ri_tol: float = 20.0,
    feature_prefix: str = "F",
    area_agg: str = "max",         # "max" o "sum"
    min_cosine: float = 0.80,
    mz_tol: float = 0.1,
    max_dlog10_area: float = 1.0,
    # pesi dello score (modificabili)
    w_rt: float = 1.0,
    w_ri: float = 1.0,
    w_cos: float = 1.0,
    w_dlog: float = 1.0,
) -> List[Feature]:
    if area_agg not in ("max", "sum"):
        raise ValueError("area_agg deve essere 'max' oppure 'sum'")

    comps_sorted = sorted(compounds, key=lambda x: x.rt)

    clusters: List[Dict] = []
    for c in comps_sorted:
        best_idx = None
        best_score = None

        for i, cl in enumerate(clusters):
            s = _score_candidate(
                c, cl,
                rt_tol=rt_tol,
                use_ri=use_ri,
                ri_tol=ri_tol,
                min_cosine=min_cosine,
                mz_tol=mz_tol,
                max_dlog10_area=max_dlog10_area,
                w_rt=w_rt, w_ri=w_ri, w_cos=w_cos, w_dlog=w_dlog,
            )
            if s is None:
                continue
            if best_score is None or s < best_score:
                best_score = s
                best_idx = i

        if best_idx is None:
            clusters.append({
                "members": [c],
                "rt_ref": c.rt,
                "ri_ref": c.ri,
                "rep": c,
            })
        else:
            cl = clusters[best_idx]
            cl["members"].append(c)

            # aggiorno riferimenti pesati
            _update_refs_weighted(cl)

            # aggiorno representative (area*purity più alta)
            rep = cl["rep"]
            if _rep_quality(c) > _rep_quality(rep):
                cl["rep"] = c

    # costruisco le Feature
    features: List[Feature] = []
    for k, cl in enumerate(clusters, start=1):
        feature_id = f"{feature_prefix}{k:06d}"

        # aggregazione per sample
        by_sample: Dict[str, float] = {}
        for m in cl["members"]:
            sid = m.sample_id
            if sid not in by_sample:
                by_sample[sid] = m.area
            else:
                by_sample[sid] = max(by_sample[sid], m.area) if area_agg == "max" else (by_sample[sid] + m.area)

        features.append(
            Feature(
                feature_id=feature_id,
                rt_ref=cl["rt_ref"],
                ri_ref=cl.get("ri_ref", None),
                by_sample_area=by_sample
            )
        )

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

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import Compound

@dataclass
class Feature:
    feature_id: str
    rt_ref: float  # RT di riferimento (minuti)
    by_sample_area: Dict[str, float]  # sample_id -> area (0 se assente)

def _assign_to_cluster(
    rt: float,
    clusters: List[Dict],
    rt_tol: float
) -> Optional[int]:
    """
    Trova l'indice del cluster compatibile (RT entro rt_tol).
    Sceglie il cluster con distanza minima.
    """
    best_idx = None
    best_dist = None
    for i, cl in enumerate(clusters):
        dist = abs(rt - cl["rt_ref"])
        if dist <= rt_tol:
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
    return best_idx

def align_compounds_rt_only(
    compounds: List[Compound],
    *,
    rt_tol: float = 1.0,          # <-- DEFAULT che mi hai chiesto (minuti)
    feature_prefix: str = "F",
    area_agg: str = "max",        # "max" oppure "sum"
) -> List[Feature]:
    """
    Allineamento 'baseline' basato SOLO su RT.
    - Raggruppa i Compound in feature se |rt - rt_ref| <= rt_tol
    - Per ciascuna feature produce area per sample (max o sum se più compound cadono nello stesso cluster)

    Ritorna una lista di Feature.
    """
    if area_agg not in ("max", "sum"):
        raise ValueError("area_agg deve essere 'max' oppure 'sum'")

    # Ordino per RT per avere clustering stabile
    comps_sorted = sorted(compounds, key=lambda c: c.rt)

    # Ogni cluster = dict con rt_ref e membri
    clusters: List[Dict] = []
    for c in comps_sorted:
        idx = _assign_to_cluster(c.rt, clusters, rt_tol)
        if idx is None:
            clusters.append({"rt_ref": c.rt, "members": [c]})
        else:
            clusters[idx]["members"].append(c)
            # aggiorno rt_ref come media dei membri (semplice e stabile)
            rts = [m.rt for m in clusters[idx]["members"]]
            clusters[idx]["rt_ref"] = sum(rts) / len(rts)

    # Costruisco le Feature
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
                else:  # sum
                    by_sample[sid] = by_sample[sid] + m.area

        features.append(Feature(feature_id=feature_id, rt_ref=rt_ref, by_sample_area=by_sample))

    return features

def features_to_table(features: List[Feature], *, fill_missing: float = 0.0):
    """
    Converte le Feature in una tabella.
    Se pandas è disponibile, ritorna un DataFrame.
    Altrimenti ritorna (header, rows) in puro Python.
    """
    # lista ordinata dei sample_id presenti
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
        df = pd.DataFrame(rows, columns=header)
        return df
    except Exception:
        return header, rows

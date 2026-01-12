from __future__ import annotations
from typing import Dict, List, Optional

from .models import Compound
from .alignment import Cluster, update_cluster_after_adding, rank_candidates_for_feature

def rescue_core_missing(
    clusters: List[Cluster],
    compounds_by_sample: Dict[str, List[Compound]],
    *,
    core_feature_ids: List[str],
    sample_ids: List[str],
    area_agg: str,
    rt_tol: float,
    use_ri: bool,
    ri_tol: float,
    min_cosine: float,
    mz_tol: float,
    max_dlog10_area: float,
    max_rescue_score: Optional[float] = 1.2,     # None = disattiva
    min_score_margin: Optional[float] = 0.15,    # None = disattiva
) -> Dict[str, List[str]]:
    rescued: Dict[str, List[str]] = {}

    for cl in clusters:
        if cl.feature_id not in core_feature_ids:
            continue

        for sid in sample_ids:
            if sid in cl.by_sample_area and cl.by_sample_area[sid] > 0:
                continue

            candidates = compounds_by_sample.get(sid, [])
            if not candidates:
                continue

            ranked = rank_candidates_for_feature(
                candidates,
                rt_ref=cl.rt_ref,
                ri_ref=cl.ri_ref,
                rep=cl.rep,
                rt_tol=rt_tol,
                use_ri=use_ri,
                ri_tol=ri_tol,
                min_cosine=min_cosine,
                mz_tol=mz_tol,
                max_dlog10_area=max_dlog10_area,
            )
            if not ranked:
                continue

            best, best_score = ranked[0]

            if max_rescue_score is not None and best_score > max_rescue_score:
                continue

            if min_score_margin is not None and len(ranked) >= 2:
                second_score = ranked[1][1]
                if (second_score - best_score) < min_score_margin:
                    continue

            cl.members.append(best)
            update_cluster_after_adding(cl, area_agg=area_agg)
            rescued.setdefault(cl.feature_id, []).append(sid)

    return rescued

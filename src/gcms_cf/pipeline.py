from __future__ import annotations
from pathlib import Path

from .io_readers import read_centroid_csv, read_centroid_csv_with_ri
from .deconvolution import toy_deconvolve_rt_peaks
from .alignment import align_compounds_clusters, clusters_to_table
from .export_xlsx import export_feature_table
from .core_compounds import add_core_flags_df
from .rescue import rescue_core_missing

def run_demo_pipeline(
    centroid_csv_path: str | Path,
    out_xlsx_path: str | Path,
    *,
    # STRICT (pass 1)
    rt_tol: float = 1.0,
    use_ri: bool = False,
    ri_tol: float = 20.0,
    area_agg: str = "max",
    sheet_name: str = "FeatureTable",
    min_cosine: float = 0.80,
    mz_tol: float = 0.1,
    max_dlog10_area: float = 1.0,
    # CORE
    core_frac: float | None = None,   # es. 0.80
    # RELAX (pass 2)
    relax_core: bool = False,
    relax_rt_tol: float | None = None,
    relax_ri_tol: float | None = None,
    relax_min_cosine: float | None = None,
    relax_max_dlog10_area: float | None = None,
    # NUOVI FRENI RESCUE
    max_rescue_score: float | None = 1.2,     # None = disattiva
    min_score_margin: float | None = 0.15,    # None = disattiva
) -> Path:
    centroid_csv_path = Path(centroid_csv_path)
    out_xlsx_path = Path(out_xlsx_path)

    # Lettura (con o senza RI)
    if use_ri:
        peaks, ri_map = read_centroid_csv_with_ri(centroid_csv_path)
    else:
        peaks = read_centroid_csv(centroid_csv_path)
        ri_map = None

    sample_ids = sorted(list(peaks.keys()))

    # Deconvoluzione toy -> Compound
    all_compounds = []
    for sample_id, peaks_by_rt in peaks.items():
        ri_by_rt = ri_map.get(sample_id, {}) if ri_map is not None else None
        comps = toy_deconvolve_rt_peaks(peaks_by_rt, sample_id=sample_id, ri_by_rt=ri_by_rt)
        all_compounds.extend(comps)

    # indicizzazione per sample (serve alla rescue)
    compounds_by_sample = {}
    for c in all_compounds:
        compounds_by_sample.setdefault(c.sample_id, []).append(c)

    # PASS 1: STRICT
    clusters_strict = align_compounds_clusters(
        all_compounds,
        rt_tol=rt_tol,
        use_ri=use_ri,
        ri_tol=ri_tol,
        area_agg=area_agg,
        min_cosine=min_cosine,
        mz_tol=mz_tol,
        max_dlog10_area=max_dlog10_area,
    )

    df_strict = clusters_to_table(clusters_strict, fill_missing=0.0)

    if core_frac is None:
        export_feature_table(df_strict, out_xlsx_path, sheet_name=sheet_name)
        return out_xlsx_path

    # Core flags su STRICT
    df_strict2, sample_cols = add_core_flags_df(df_strict, core_frac=core_frac)
    core_ids = list(df_strict2[df_strict2["is_core"] == True]["feature_id"].astype(str))

    # PASS 2: RELAX solo sui core
    rescued_map = {}
    if relax_core and core_ids:
        rt_tol_rel = relax_rt_tol if relax_rt_tol is not None else (rt_tol * 1.5)
        ri_tol_rel = relax_ri_tol if relax_ri_tol is not None else (ri_tol * 1.5)
        min_cos_rel = relax_min_cosine if relax_min_cosine is not None else max(min_cosine - 0.05, 0.50)
        dlog_rel = relax_max_dlog10_area if relax_max_dlog10_area is not None else (max_dlog10_area + 0.5)

        rescued_map = rescue_core_missing(
            clusters_strict,
            compounds_by_sample,
            core_feature_ids=core_ids,
            sample_ids=sample_ids,
            area_agg=area_agg,
            rt_tol=rt_tol_rel,
            use_ri=use_ri,
            ri_tol=ri_tol_rel,
            min_cosine=min_cos_rel,
            mz_tol=mz_tol,
            max_dlog10_area=dlog_rel,
            max_rescue_score=max_rescue_score,
            min_score_margin=min_score_margin,
        )

    # Tabella finale (dopo rescue)
    import pandas as pd  # type: ignore

    df_final = clusters_to_table(clusters_strict, fill_missing=0.0)

    df_final = df_final.merge(
        df_strict2[["feature_id", "n_present", "presence_frac", "is_core"]],
        on="feature_id",
        how="left",
    ).rename(columns={"n_present": "n_present_strict", "presence_frac": "presence_frac_strict"})

    present_final = (df_final[sample_cols] > 0.0)
    df_final["n_present_final"] = present_final.sum(axis=1)
    df_final["presence_frac_final"] = df_final["n_present_final"] / float(len(sample_cols))

    df_final["n_rescued"] = df_final["feature_id"].map(lambda fid: len(rescued_map.get(fid, []))).fillna(0).astype(int)
    df_final["rescued_samples"] = df_final["feature_id"].map(lambda fid: ";".join(rescued_map.get(fid, [])) if fid in rescued_map else "")

    df_final = df_final.sort_values(
        ["is_core", "presence_frac_final", "rt_ref"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    export_feature_table(df_final, out_xlsx_path, sheet_name=sheet_name)
    return out_xlsx_path

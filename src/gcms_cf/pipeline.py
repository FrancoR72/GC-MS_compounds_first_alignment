from __future__ import annotations
from pathlib import Path

from .io_readers import read_centroid_csv, read_centroid_csv_with_ri
from .deconvolution import toy_deconvolve_rt_peaks
from .alignment import align_compounds_rt_only, features_to_table
from .export_xlsx import export_feature_table
from .core_compounds import add_core_flags_df

def run_demo_pipeline(
    centroid_csv_path: str | Path,
    out_xlsx_path: str | Path,
    *,
    rt_tol: float = 1.0,
    use_ri: bool = False,
    ri_tol: float = 20.0,
    area_agg: str = "max",
    sheet_name: str = "FeatureTable",
    min_cosine: float = 0.80,
    mz_tol: float = 0.1,
    max_dlog10_area: float = 1.0,
    # NUOVO: core compounds
    core_frac: float | None = None,     # es. 0.80 per core “quasi tutti”
) -> Path:
    """
    Pipeline baseline:
    CSV centroid -> compounds (toy) -> alignment -> tabella -> export XLSX

    Se core_frac è impostato (es. 0.80):
      aggiunge colonne n_present / presence_frac / is_core e promuove i core in alto.
    """
    centroid_csv_path = Path(centroid_csv_path)
    out_xlsx_path = Path(out_xlsx_path)

    if use_ri:
        peaks, ri_map = read_centroid_csv_with_ri(centroid_csv_path)
    else:
        peaks = read_centroid_csv(centroid_csv_path)
        ri_map = None

    all_compounds = []
    for sample_id, peaks_by_rt in peaks.items():
        ri_by_rt = ri_map.get(sample_id, {}) if ri_map is not None else None
        comps = toy_deconvolve_rt_peaks(peaks_by_rt, sample_id=sample_id, ri_by_rt=ri_by_rt)
        all_compounds.extend(comps)

    features = align_compounds_rt_only(
        all_compounds,
        rt_tol=rt_tol,
        use_ri=use_ri,
        ri_tol=ri_tol,
        area_agg=area_agg,
        min_cosine=min_cosine,
        mz_tol=mz_tol,
        max_dlog10_area=max_dlog10_area,
    )

    table = features_to_table(features, fill_missing=0.0)

    # Se è un DataFrame e core_frac è richiesto, aggiungo flags core
    try:
        import pandas as pd  # type: ignore
        if core_frac is not None and isinstance(table, pd.DataFrame):
            table, _ = add_core_flags_df(table, core_frac=core_frac)
    except Exception:
        pass

    export_feature_table(table, out_xlsx_path, sheet_name=sheet_name)
    return out_xlsx_path

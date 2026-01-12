from __future__ import annotations
from pathlib import Path
from typing import Optional

from .io_readers import read_centroid_csv
from .deconvolution import toy_deconvolve_rt_peaks
from .alignment import align_compounds_rt_only, features_to_table
from .export_xlsx import export_feature_table

def run_demo_pipeline(
    centroid_csv_path: str | Path,
    out_xlsx_path: str | Path,
    *,
    rt_tol: float = 1.0,          # default richiesto: 1 minuto
    area_agg: str = "max",        # "max" o "sum"
    sheet_name: str = "FeatureTable",
) -> Path:
    """
    Pipeline baseline (demo):
    CSV centroid -> compounds (toy) -> alignment RT-only -> tabella -> export XLSX

    Parametri modificabili dall'utente:
    - rt_tol: tolleranza RT (minuti)
    - area_agg: aggregazione aree se più compound dello stesso sample finiscono nella stessa feature
    """
    centroid_csv_path = Path(centroid_csv_path)
    out_xlsx_path = Path(out_xlsx_path)

    data = read_centroid_csv(centroid_csv_path)

    all_compounds = []
    for sample_id, peaks_by_rt in data.items():
        comps = toy_deconvolve_rt_peaks(peaks_by_rt, sample_id=sample_id)
        all_compounds.extend(comps)

    features = align_compounds_rt_only(all_compounds, rt_tol=rt_tol, area_agg=area_agg)
    table = features_to_table(features, fill_missing=0.0)

    export_feature_table(table, out_xlsx_path, sheet_name=sheet_name)
    return out_xlsx_path

from __future__ import annotations
from typing import List, Tuple

def add_core_flags_df(df, *, core_frac: float = 0.67, presence_threshold: float = 0.0):
    """
    Aggiunge colonne per identificare i 'core compounds' su una feature table (DataFrame).

    Regola presenza:
      una feature è presente in un campione se area > presence_threshold (default 0.0)

    Colonne aggiunte:
      - n_present
      - presence_frac
      - is_core  (presence_frac >= core_frac)

    Ritorna:
      df_annotated (ordinato con i core in alto), sample_cols
    """
    # colonne campioni = tutte tranne le colonne di meta
    meta_cols = {"feature_id", "rt_ref", "ri_ref"}
    sample_cols: List[str] = [c for c in df.columns if c not in meta_cols]

    if len(sample_cols) == 0:
        df["n_present"] = 0
        df["presence_frac"] = 0.0
        df["is_core"] = False
        return df, sample_cols

    present = (df[sample_cols] > presence_threshold)
    df["n_present"] = present.sum(axis=1)
    df["presence_frac"] = df["n_present"] / float(len(sample_cols))
    df["is_core"] = df["presence_frac"] >= float(core_frac)

    # Promozione: core prima, poi per presence_frac decrescente, poi per rt_ref
    sort_cols = ["is_core", "presence_frac", "rt_ref"]
    sort_asc = [False, False, True]

    df_sorted = df.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
    return df_sorted, sample_cols

def core_feature_ids(df, *, core_frac: float = 0.67) -> List[str]:
    """Ritorna la lista di feature_id core, assumendo che df abbia già le colonne presence_frac/is_core oppure ricalcolandole."""
    if "is_core" in df.columns:
        core_df = df[df["is_core"] == True]
        return list(core_df["feature_id"].astype(str))
    # fallback minimale (se manca is_core)
    if "presence_frac" in df.columns:
        core_df = df[df["presence_frac"] >= float(core_frac)]
        return list(core_df["feature_id"].astype(str))
    return []

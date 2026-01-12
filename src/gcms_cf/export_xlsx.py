from __future__ import annotations
from pathlib import Path
from typing import Any

def export_feature_table(table: Any, out_xlsx_path: str | Path, *, sheet_name: str = "FeatureTable") -> Path:
    """
    Esporta una feature table in XLSX.
    - Se 'table' è un pandas DataFrame, lo salva direttamente.
    - Se è una tupla (header, rows), la converte in DataFrame e salva.

    Ritorna il Path del file scritto.
    """
    out_xlsx_path = Path(out_xlsx_path)
    out_xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd  # in Colab di solito è già presente

    if hasattr(table, "to_excel"):
        df = table
    else:
        header, rows = table
        df = pd.DataFrame(rows, columns=header)

    df.to_excel(out_xlsx_path, index=False, sheet_name=sheet_name)
    return out_xlsx_path

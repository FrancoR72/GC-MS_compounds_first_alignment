from __future__ import annotations
import csv
from typing import Dict, List, Tuple
from pathlib import Path

from .models import Peak

def read_centroid_csv(csv_path: str | Path) -> Dict[str, Dict[float, List[Peak]]]:
    """
    Legge un CSV con colonne: sample_id, rt_min, mz, intensity
    Ritorna: dict[sample_id][rt_min] -> lista di Peak
    """
    csv_path = Path(csv_path)
    out: Dict[str, Dict[float, List[Peak]]] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "rt_min", "mz", "intensity"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV deve contenere colonne {sorted(required)}. Trovate: {reader.fieldnames}")

        for row in reader:
            sample_id = row["sample_id"]
            rt = float(row["rt_min"])
            mz = float(row["mz"])
            intensity = float(row["intensity"])

            out.setdefault(sample_id, {}).setdefault(rt, []).append(Peak(mz=mz, intensity=intensity))

    return out

def read_centroid_csv_with_ri(csv_path: str | Path) -> Tuple[Dict[str, Dict[float, List[Peak]]], Dict[str, Dict[float, float]]]:
    """
    Legge un CSV con colonne: sample_id, rt_min, ri, mz, intensity
    Ritorna:
      - peaks: dict[sample_id][rt_min] -> lista di Peak
      - ri_map: dict[sample_id][rt_min] -> ri
    """
    csv_path = Path(csv_path)
    peaks: Dict[str, Dict[float, List[Peak]]] = {}
    ri_map: Dict[str, Dict[float, float]] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "rt_min", "ri", "mz", "intensity"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV con RI deve contenere colonne {sorted(required)}. Trovate: {reader.fieldnames}")

        for row in reader:
            sample_id = row["sample_id"]
            rt = float(row["rt_min"])
            ri = float(row["ri"])
            mz = float(row["mz"])
            intensity = float(row["intensity"])

            peaks.setdefault(sample_id, {}).setdefault(rt, []).append(Peak(mz=mz, intensity=intensity))

            ri_map.setdefault(sample_id, {})
            if rt in ri_map[sample_id]:
                # se ripetuto, deve essere coerente
                if abs(ri_map[sample_id][rt] - ri) > 1e-6:
                    raise ValueError(f"RI incoerente per {sample_id} RT={rt}: {ri_map[sample_id][rt]} vs {ri}")
            else:
                ri_map[sample_id][rt] = ri

    return peaks, ri_map

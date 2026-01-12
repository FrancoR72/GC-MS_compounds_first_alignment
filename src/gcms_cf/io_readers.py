from __future__ import annotations
import csv
from typing import Dict, List
from pathlib import Path

from .models import Peak

def read_centroid_csv(csv_path: str | Path) -> Dict[str, Dict[float, List[Peak]]]:
    """
    Legge un CSV con colonne: sample_id, rt_min, mz, intensity
    Restituisce: dict[sample_id][rt_min] -> lista di Peak
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

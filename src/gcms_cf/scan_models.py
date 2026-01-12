from __future__ import annotations
from dataclasses import dataclass
from typing import List

from .models import Peak

@dataclass(frozen=True)
class Scan:
    scan_id: int
    rt_min: float
    peaks: List[Peak]

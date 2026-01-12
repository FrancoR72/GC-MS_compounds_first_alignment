from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class Peak:
    mz: float
    intensity: float

@dataclass(frozen=True)
class Compound:
    compound_id: str          # id interno (es. "C000123")
    sample_id: str            # nome/codice campione
    rt: float                 # retention time (secondi o minuti: lo decidiamo dopo)
    ri: Optional[float]       # retention index (se disponibile)
    area: float               # area del compound
    purity: float             # 0..1 o 0..100 (lo decidiamo dopo)
    spectrum: List[Peak]      # lista di picchi (fingerprint)

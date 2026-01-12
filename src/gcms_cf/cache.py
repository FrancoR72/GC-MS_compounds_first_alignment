from pathlib import Path
import pickle
import hashlib

def _key_to_filename(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"{h}.pkl"

def cache_get(cache_dir: str | Path, key: str):
    cache_dir = Path(cache_dir)
    f = cache_dir / _key_to_filename(key)
    if not f.exists():
        return None
    with open(f, "rb") as fp:
        return pickle.load(fp)

def cache_set(cache_dir: str | Path, key: str, value):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / _key_to_filename(key)
    with open(f, "wb") as fp:
        pickle.dump(value, fp)

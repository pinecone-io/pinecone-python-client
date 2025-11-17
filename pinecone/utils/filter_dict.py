from typing import Dict


def filter_dict(d: Dict, allowed_keys: tuple[str, ...]) -> Dict:
    return {k: v for k, v in d.items() if k in allowed_keys}

def filter_dict(d: dict, allowed_keys: tuple[str, ...]) -> dict:
    return {k: v for k, v in d.items() if k in allowed_keys}

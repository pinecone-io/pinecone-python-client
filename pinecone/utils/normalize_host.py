def normalize_host(host: str | None) -> str:
    if host is None:
        return ""
    if host.startswith("https://"):
        return host
    if host.startswith("http://"):
        return host
    return "https://" + host

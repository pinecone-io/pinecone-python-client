def get_version():
    try:
        with open("pyproject.toml", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return "unknown"

    version_line = next((line for line in lines if line.startswith("version")), None)
    if version_line:
        # Extract the version number from the line
        version = version_line.split("=")[1].strip().strip('"')
        return version
    else:
        return "unknown"

__version__ = get_version()
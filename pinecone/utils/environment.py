from pathlib import Path

def get_version():
    return Path(__file__).parent.parent.joinpath("__version__").read_text().strip()


def get_environment():
    return Path(__file__).parent.parent.joinpath("__environment__").read_text().strip()
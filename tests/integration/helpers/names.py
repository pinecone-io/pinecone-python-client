import uuid
# import hashlib


def generate_name(test_name: str, label: str, max_length: int = 20) -> str:
    """
    The goal of this function is to produce names that are unique across the
    test suite but deterministic when the test is run multiple times, when ordering
    changes, different subsets of tests are run, etc.

    To accomodate this, we hash the test name and label. We truncate the hexdigest
    since the full length of 64 characters exceeds the allowed length of some fields
    in the API. For example, index names must be 45 characters or less.
    """
    # return hashlib.sha256(f"{test_name}-{label}".encode()).hexdigest()[:max_length]

    # Having names be fully deterministic led to problems when multiple test builds
    # are running in parallel, for example running the same tests in parallel for
    # different python versions. We can solve this by incorporating more information
    # into the name generation, but for now as a quick fix to unblock a release we
    # will just fall back to using a random uuid.
    return str(uuid.uuid4())

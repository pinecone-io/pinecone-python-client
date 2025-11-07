"""Setup functions for weird_ids integration tests."""

from tests.integration.helpers import embedding_values, poll_until_lsn_reconciled
import itertools
import logging

logger = logging.getLogger(__name__)


def weird_invalid_ids():
    """Returns a list of invalid vector IDs that should be rejected by the API."""
    invisible = [
        "\u2800",  # U+2800
        "\u00a0",  # U+00A0
        "\u00ad",  # U+00AD
        "\u17f4",  # U+17F4
        "\u180e",  # U+180E
        "\u2000",  # U+2000
        "\u2001",  # U+2001
        "\u2002",  # U+2002
    ]
    emojis = list("🌲🍦")
    two_byte = list("田中さんにあげて下さい")
    quotes = [
        "\u2018",  # '
        "\u2019",  # '
        "\u201c",  # "
        "\u201d",  # "
        "\u201e",  # „
        "\u201f",  # ‟
        "\u2039",  # ‹
        "\u203a",  # ›
        "\u275b",  # ❛
        "\u275c",  # ❜
        "\u275d",  # ❝
        "\u275e",  # ❞
        "\u276e",  # ❮
        "\u276f",  # ❯
        "\uff02",  # ＂
        "\uff07",  # ＇
        "\uff62",  # ｢
        "\uff63",  # ｣
    ]

    return invisible + emojis + two_byte + quotes


def weird_valid_ids():
    """Returns a list of valid but unusual vector IDs for testing edge cases.

    Drawing inspiration from the big list of naughty strings:
    https://github.com/minimaxir/big-list-of-naughty-strings/blob/master/blns.txt
    """
    ids = []

    numbers = list("1234567890")
    invisible = [" ", "\n", "\t", "\r"]
    punctuation = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    escaped = [f"\\{c}" for c in punctuation]

    characters = numbers + invisible + punctuation + escaped
    ids.extend(characters)
    ids.extend(["".join(x) for x in itertools.combinations_with_replacement(characters, 2)])

    boolean_ish = [
        "undefined",
        "nil",
        "null",
        "Null",
        "NULL",
        "None",
        "True",
        "False",
        "true",
        "false",
    ]
    ids.extend(boolean_ish)

    script_injection = [
        "<script>alert(0)</script>",
        "<svg><script>123<1>alert(3)</script>",
        '" onfocus=JaVaSCript:alert(10) autofocus',
        "javascript:alert(1)",
        "javascript:alert(1);",
        '<img src\x32=x onerror="javascript:alert(182)">1;DROP TABLE users',
        "' OR 1=1 -- 1",
        "' OR '1'='1",
    ]
    ids.extend(script_injection)

    unwanted_interpolation = ["$HOME", "$ENV{'HOME'}", "%d", "%s", "%n", "%x", "{0}"]
    ids.extend(unwanted_interpolation)

    return ids


def setup_weird_ids_data(idx, target_namespace, wait):
    """Upsert vectors with weird IDs for testing.

    Args:
        idx: Index instance to upsert to
        target_namespace: Namespace to upsert vectors to
        wait: Whether to wait for LSN reconciliation
    """
    weird_ids = weird_valid_ids()
    batch_size = 100
    for i in range(0, len(weird_ids), batch_size):
        chunk = weird_ids[i : i + batch_size]
        upsert1 = idx.upsert(
            vectors=[(x, embedding_values(2)) for x in chunk], namespace=target_namespace
        )

        chunk_response_info = upsert1._response_info
        last_response_info = chunk_response_info

    if wait:
        poll_until_lsn_reconciled(idx, last_response_info, namespace=target_namespace)

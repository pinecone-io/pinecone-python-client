from ..helpers import poll_fetch_for_ids_in_namespace
from pinecone import Vector
from .utils import embedding_values
import itertools


def setup_data(idx, target_namespace, wait):
    # Upsert without metadata
    idx.upsert(
        vectors=[("1", embedding_values(2)), ("2", embedding_values(2)), ("3", embedding_values(2))],
        namespace=target_namespace,
    )

    # Upsert with metadata
    idx.upsert(
        vectors=[
            Vector(id="4", values=embedding_values(2), metadata={"genre": "action", "runtime": 120}),
            Vector(id="5", values=embedding_values(2), metadata={"genre": "comedy", "runtime": 90}),
            Vector(id="6", values=embedding_values(2), metadata={"genre": "romance", "runtime": 240}),
        ],
        namespace=target_namespace,
    )

    # Upsert with dict
    idx.upsert(
        vectors=[
            {"id": "7", "values": embedding_values(2)},
            {"id": "8", "values": embedding_values(2)},
            {"id": "9", "values": embedding_values(2)},
        ],
        namespace=target_namespace,
    )

    if wait:
        poll_fetch_for_ids_in_namespace(
            idx, ids=["1", "2", "3", "4", "5", "6", "7", "8", "9"], namespace=target_namespace
        )


def setup_list_data(idx, target_namespace, wait):
    # Upsert a bunch more stuff for testing list pagination
    for i in range(0, 1000, 50):
        idx.upsert(vectors=[(str(i + d), embedding_values(2)) for d in range(50)], namespace=target_namespace)

    if wait:
        poll_fetch_for_ids_in_namespace(idx, ids=["999"], namespace=target_namespace)


def weird_invalid_ids():
    invisible = [
        "⠀",  # U+2800
        " ",  # U+00A0
        "­",  # U+00AD
        "឴",  # U+17F4
        "᠎",  # U+180E
        " ",  # U+2000
        " ",  # U+2001
        " ",  # U+2002
    ]
    emojis = list("🌲🍦")
    two_byte = list("田中さんにあげて下さい")
    quotes = ["‘", "’", "“", "”", "„", "‟", "‹", "›", "❛", "❜", "❝", "❞", "❮", "❯", "＂", "＇", "｢", "｣"]

    return invisible + emojis + two_byte + quotes


def weird_valid_ids():
    # Drawing inspiration from the big list of naughty strings https://github.com/minimaxir/big-list-of-naughty-strings/blob/master/blns.txt
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
        '<img src\x32=x onerror="javascript:alert(182)">' "1;DROP TABLE users",
        "' OR 1=1 -- 1",
        "' OR '1'='1",
    ]
    ids.extend(script_injection)

    unwanted_interpolation = [
        "$HOME",
        "$ENV{'HOME'}",
        "%d",
        "%s",
        "%n",
        "%x",
        "{0}",
    ]
    ids.extend(unwanted_interpolation)

    return ids


def setup_weird_ids_data(idx, target_namespace, wait):
    weird_ids = weird_valid_ids()
    batch_size = 100
    for i in range(0, len(weird_ids), batch_size):
        chunk = weird_ids[i : i + batch_size]
        idx.upsert(vectors=[(x, embedding_values(2)) for x in chunk], namespace=target_namespace)

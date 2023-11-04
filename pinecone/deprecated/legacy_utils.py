import re

try:
    from google.protobuf.struct_pb2 import Struct
    from google.protobuf import json_format
except Exception:
    pass  # ignore for non-[grpc] installations

DNS_COMPATIBLE_REGEX = re.compile("^[a-z0-9]([a-z0-9]|[-])+[a-z0-9]$")

def validate_dns_name(name):
    if not DNS_COMPATIBLE_REGEX.match(name):
        raise ValueError(
            "{} is invalid - service names and node names must consist of lower case "
            "alphanumeric characters or '-', start with an alphabetic character, and end with an "
            "alphanumeric character (e.g. 'my-name', or 'abc-123')".format(name)
        )


def proto_struct_to_dict(s: "Struct") -> dict:
    return json_format.MessageToDict(s)

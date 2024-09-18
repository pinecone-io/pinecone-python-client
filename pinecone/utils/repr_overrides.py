import json
from datetime import datetime


def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)


def install_json_repr_override(klass):
    klass.__repr__ = lambda self: json.dumps(self.to_dict(), indent=4, sort_keys=False, default=custom_serializer)

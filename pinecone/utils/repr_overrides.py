import json


def install_json_repr_override(klass):
    klass.__repr__ = lambda self: json.dumps(self.to_dict(), indent=4, sort_keys=False)

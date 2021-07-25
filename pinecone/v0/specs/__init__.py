#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

import yaml
import json
import abc


class Spec:
    """
    Object that can be deployed to a controller
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_obj(cls, obj: dict) -> "Spec":
        raise NotImplementedError

    @abc.abstractmethod
    def to_obj(self) -> dict:
        raise NotImplementedError

    def validate(self):
        pass

    @classmethod
    def from_json(cls, spec: str) -> "Spec":
        """Creates class instance from json string."""
        obj = None
        try:
            obj = json.loads(spec)
        except Exception as e:
            raise Exception(f"Unable to load spec: {e}")
        return cls.from_obj(obj)

    def to_json(self) -> str:
        return json.dumps(self.to_obj())

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_obj())

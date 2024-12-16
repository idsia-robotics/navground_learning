from __future__ import annotations
from collections.abc import Mapping

from typing import Any
try:
    from typing import Self
except ImportError:
    try:
        from typing_extensions import Self
    except ImportError:
        pass


class Registrable:

    _type: str = ''

    @property
    def asdict(self) -> dict[str, Any]:
        return dict(**self.get_dict(), type=self._type)

    def get_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def make_from_dict(cls, value: Mapping[str, Any]) -> Self:
        return cls()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if 'type' not in value:
            raise ValueError(f"{cls.__name__}: type not provided")
        t = value.get('type')
        for subcls in cls.__subclasses__():
            if subcls._type == t:
                vs = {k: v for k, v in value.items() if k != 'type'}
                return subcls.make_from_dict(vs)
        raise ValueError(f"{cls.__name__}: type {value['type']} not registered")

    # Renamed to `register_name` because `name` conflicts with Python3.10 ABC
    def __init_subclass__(cls, register_name: str = '') -> None:
        cls._type = register_name

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class Registrable:
    """
    Internal helper class that can be converted to/from a plain dictionary
    """

    _type: str = ''

    @property
    def asdict(self) -> dict[str, Any]:
        """
        The YAML-able dictionary representation.

        Adds the class type name to the dictionary returned by
        :py:meth:`_get_dict`.

        :returns: A YAML-able dictionary
        """
        return dict(**self._get_dict(), type=self._type)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """
        Load an instance from a dictionary.

        Selects a class from field ``type`` and calls its method
        :py:meth:`_make_from_dict`.

        :param      value:  The value

        :raises ValueError: If ``type`` is not defined or not registered.

        :returns:   An instance initialized from the content of ``value``.
        """
        if 'type' not in value:
            raise ValueError(f"{cls.__name__}: type not provided")
        t = value.get('type')
        for subcls in cls.__subclasses__():
            if subcls._type == t:
                vs = {k: v for k, v in value.items() if k != 'type'}
                return subcls._make_from_dict(vs)
        raise ValueError(
            f"{cls.__name__}: type {value['type']} not registered")

    def _get_dict(self) -> dict[str, Any]:
        """
        Sub-classes can override this method to
        provide fields in :py:data:`asdict`.

        The default implementation returns an empty dict.

        :returns: A YAML-able dictionary
        """
        return {}

    @classmethod
    def _make_from_dict(cls, value: Mapping[str, Any]) -> Self:
        """
        Sub-classes can override this method to load an instance from
        a dictionary.

        The default implementation initializes the class with no arguments.

        :param      value:  The value

        :returns:   An instance initialized from the content of ``value``.
        """
        return cls()

    # Renamed to `register_name` because `name` conflicts with Python3.10 ABC
    def __init_subclass__(cls, register_name: str = '') -> None:
        cls._type = register_name

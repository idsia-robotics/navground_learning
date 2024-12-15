from __future__ import annotations

from collections.abc import Sequence, Mapping
from enum import IntEnum
from operator import itemgetter
from typing import Any, Literal, TypeAlias, TypeVar, cast

T = TypeVar("T")


class Indices:
    """
    A union type that can be used to get a part of a sequence or
    of a dictionary with integer keys.

    - :py:attr:`Indices.all` -> select all items
    - :py:class:`set` -> select the items with indices in the set
    - :py:class:`slice` -> select the items covered by the slide.

    For example:

    >>> Indices.all().sub_sequence([1, 2, 3])
    [1, 2, 3]
    >>> Indices.all().sub_dict({1: 2, 3: 4, 5: 6})
    {1: 2, 3: 4, 5: 6}
    >>> Indices({0, 2}).sub_sequence((1, 2, 3))
    (1, 3)
    >>> Indices({0, 2}).sub_dict({1: 2, 3: 4, 5: 6})
    {}
    >>> Indices(slice(-2, None)).sub_sequence((1, 2, 3, 4, 5))
    (4, 5)
    >>> Indices(slice(1, 3)).sub_dict({1: 2, 3: 4, 5: 6})
    {1: 2}

    :param      value:  The value
        - ``"ALL"``: initializes :py:attr:`Indices.all`
        - :py:class:`set`: initializes the indices from the set
        - :py:class:`slice`: initializes the indices from the slide.
        - :py:class:`Indices`: copy the value
    """

    __slots__ = '_type', '_value'

    class Type(IntEnum):
        slice = 0
        set = 1
        all = 2

    @classmethod
    def all(cls) -> Indices:
        """
        Indices that covers all the items.
        """
        return Indices()

    def __init__(self, value: IndicesLike = "ALL"):
        self._value: slice | set[int] | Literal['ALL']
        self._type: Indices.Type
        if isinstance(value, Indices):
            self._value = value._value
            self._type = value._type
        elif isinstance(value, slice):
            self._type = Indices.Type.slice
            self._value = value
        elif isinstance(value, (list, tuple, set)):
            self._type = Indices.Type.set
            self._value = set(value)
        else:
            self._type = Indices.Type.all
            self._value = "ALL"

    def __repr__(self) -> str:
        if self._type == Indices.Type.all:
            return "Indices.all()"
        return f"Indices({self._value!r})"

    def __hash__(self) -> int:
        return hash(self._value)

    def sub_sequence(self, xs: Sequence[T]) -> Sequence[T]:
        """
        Extracts a sub-sequence.

        :param      xs:  The original sequence

        :returns:   the items of xs covered by the indices
        """
        if self._type == Indices.Type.all:
            return xs
        if self._type == Indices.Type.set:
            ys = cast(set[int], self._value) & set(range(0, len(xs)))
            if ys:
                z = itemgetter(*ys)(xs)
                if len(ys) > 1:
                    return cast(Sequence[T], z)
                return type(xs)([z])  # type: ignore[call-arg]

            return type(xs)()
        return xs[cast(slice, self._value)]

    def sub_dict(self, xs: dict[int, T]) -> dict[int, T]:
        """
        Extracts a sub-set.

        :param      xs:  The original sequence

        :returns:   the items of xs covered by the indices
        """
        if self._type == Indices.Type.all:
            return xs
        return {k: v for k, v in xs.items() if k in self.as_set(max(xs) + 1)}

    def as_set(self, length: int) -> set[int]:
        """
        Covert to a set of the indices.
        Uses length to evaluate slices using :py:meth:`slice.indices`.

        :param      length:  The maximal length of the sequence

        :returns:   the set of indices
        """
        if self._type == Indices.Type.set:
            return cast(set[int], self._value)
        if self._type == Indices.Type.all:
            return set(range(0, length))
        return set(range(*cast(slice, self._value).indices(length)))

    @property
    def lowest(self) -> int | None:
        """
        The lowest index

        :returns:   The lowest index if the available, else ``None``.
        """
        if self._type == Indices.Type.all:
            return 0
        if self._type == Indices.Type.set:
            if self._value:
                return min(cast(set[int], self._value))
            return None
        a, b, _ = cast(slice, self._value).indices(1)
        if a < b:
            return a
        return None

    @property
    def asdict(self) -> dict[str, Any]:
        """
        A JSON-able representation.

        :returns:   A JSON-able dict
        """
        rs: dict[str, Any] = {}
        rs['type'] = self._type.name
        if self._type == Indices.Type.set:
            rs['values'] = list(cast(set[int], self._value))
        elif self._type == Indices.Type.slice:
            rs['start'] = cast(slice, self._value).start
            rs['stop'] = cast(slice, self._value).stop
            rs['step'] = cast(slice, self._value).step
        return rs

    @classmethod
    def from_dict(cls, rs: Mapping[str, Any]) -> Indices:
        """
        Read the indices from the representation :py:attr:`asdict`.

        :param      rs:   The dictionart

        :returns:   The indices
        :raises ValueError: if no conversion is possible
        """
        if rs['type'] == 'all':
            return cls.all()
        if rs['type'] == 'set':
            return cls(set(rs['values']))
        if rs['type'] == 'slice':
            return cls(slice(rs['start'], rs['stop'], rs['step']))
        raise ValueError(f"Could not read indices from {rs}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Indices):
            return False
        return self._value == other._value

    def intersect(self, other: Indices, length: int) -> Indices:
        """
        Intersects with another set of indices.

        :param      other:   The other set
        :param      length:  The maximal length of the sequences, used to evaluate slices.

        :returns:   The set of indices that are common to both sets.
        """
        if self == other:
            return self
        if self._type == Indices.Type.all:
            return other
        return Indices(other.as_set(length) & self.as_set(length))

    def __bool__(self) -> bool:
        """
        :returns:   True if not empty.
        """
        if self._type == Indices.Type.all:
            return True
        if self._type == Indices.Type.set:
            return bool(cast(set[int], self._value))
        s = cast(slice, self._value)
        start = s.start or 0
        step = s.step or 1
        stop = s.stop or (start + step)
        return (stop - start) * step > 0


IndicesLike: TypeAlias = Indices | slice | list[int] | tuple[int] | set[int] | Literal[
    'ALL']

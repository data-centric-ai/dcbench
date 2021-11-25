from typing import Iterator, Mapping, Optional, Sequence, Union

import pandas as pd

BASIC_TYPE = Union[int, float, str, bool]


class RowMixin:
    def __init__(self, id: str, attributes: Mapping[str, BASIC_TYPE] = None):
        self.id = id
        self._attributes = attributes

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, value):
        self._attributes = value


class Table(Mapping[str, RowMixin]):
    def __init__(self, data: Sequence[RowMixin]):
        self._data = {item.id: item for item in data}

    def __getitem__(self, k: str) -> RowMixin:
        result = self._data.get(k, None)
        if result is None:
            raise KeyError()
        return result

    def __iter__(self) -> Iterator[str]:
        return self._data.__iter__()

    def __len__(self) -> int:
        return self._data.__len__()

    @property
    def df(self):
        return pd.DataFrame.from_dict(
            {k: v.attributes for k, v in self._data.items()}, orient="index"
        )

    def __repr__(self) -> str:
        return self.df.__repr__()

    def _repr_html_(self) -> Optional[str]:
        return self.df._repr_html_()

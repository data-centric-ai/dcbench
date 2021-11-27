import copy
from itertools import chain
from typing import Dict, Iterator, Mapping, Optional, Sequence, Union

import pandas as pd

BASIC_TYPE = Union[int, float, str, bool]


class RowMixin:
    def __init__(self, id: str, attributes: Mapping[str, BASIC_TYPE] = None):
        self.id = id
        self._attributes = attributes

    @property
    def attributes(self) -> Optional[Mapping[str, BASIC_TYPE]]:
        return self._attributes

    @attributes.setter
    def attributes(self, value: Mapping[str, BASIC_TYPE]):
        self._attributes = value


class RowUnion(RowMixin):

    def __init__(self, id: str, elements: Sequence[RowMixin]):
        self._elements = elements
        attributes: Dict[str, BASIC_TYPE] = {}
        for element in reversed(elements):
            attributes.update(element.attributes)
        super().__init__(id, attributes=attributes)


def predicate(a: BASIC_TYPE, b: Union[BASIC_TYPE, slice, Sequence[BASIC_TYPE]]) -> bool:
    if isinstance(b, slice):
        return (b.start is not None and a >= b.start) and (b.stop is not None and a < b.stop)
    elif isinstance(b, Sequence):
        return a in b
    else:
        return a == b


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

    def _add_row(self, row: RowMixin) -> None:
        self._data[row.id] = row

    @property
    def df(self):
        return pd.DataFrame.from_dict(
            {k: v.attributes for k, v in self._data.items()}, orient="index"
        )

    def where(self, **kwargs: Union[BASIC_TYPE, slice, Sequence[BASIC_TYPE]]) -> "Table":
        result_data = [item for item
                       in self._data.values()
                       if all(predicate(item.attributes.get(k, None), v) for (k, v) in kwargs.items())]
        return type(self)(result_data)

    def average(self, *targets: str, groupby: Optional[Sequence[str]] = None, std: bool = False) -> "Table":
        groupby = groupby or []
        df = self.df[chain(targets, groupby)]
        if groupby is not None and len(groupby) > 0:
            df = df.groupby(groupby)
        df_result = df.mean()
        if isinstance(df_result, pd.Series):
            df_result = df_result.to_frame().T
        if std:
            df_std = df.std()
            if isinstance(df_std, pd.Series):
                df_std = df_std.to_frame().T
            df_result = pd.merge(df_result, df_std, left_index=True, right_index=True, suffixes=("", ":std"))
        df_result = df_result.reset_index()
        result_rows = [RowMixin(id=str(id), attributes=row) for id, row in df_result.iterrows()]
        return Table(result_rows)

    def __repr__(self) -> str:
        return self.df.__repr__()

    def _repr_html_(self) -> Optional[str]:
        return self.df._repr_html_()

    def __add__(self, other: RowMixin) -> "Table":
        result = copy.deepcopy(self)
        result._add_row(other)
        return result

    def __iadd__(self, other: RowMixin) -> "Table":
        self._add_row(other)
        return self

from typing import Any, Generic, Iterator, List, Mapping, Optional, TypeVar, Union

from pandas import DataFrame, Series

BUNDLE_KEY_SEPARATOR = "/"
V = TypeVar("V")


class Bundle(Generic[V], Mapping[str, Union[V, "Bundle"]]):
    def __init__(self, items: Mapping[str, V], **kwargs) -> None:
        self._items = items
        self._aux_items = dict()
        for k, v in items.items():
            splits = k.split(BUNDLE_KEY_SEPARATOR, 1)
            if len(splits) > 1:
                self._aux_items.setdefault(splits[0], {})[splits[1]] = v
        self._aux_items = dict(
            (k, type(self)[V](v, **kwargs)) for k, v in self._aux_items.items()
        )

        super().__init__()

    def __getitem__(self, k: str) -> Union[V, "Bundle"]:
        result = self._items.get(k, None)
        if result is None:
            result = self._aux_items.get(k)
        return result

    def __iter__(self) -> Iterator[str]:
        return self._items.__iter__()

    def __len__(self) -> int:
        return self._items.__len__()

    def __getattr__(self, k: str) -> Union[V, "Bundle"]:
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def get_value(obj: Any, key: str) -> Any:
    value = None
    key = key.split(".", maxsplit=1)
    if isinstance(obj, Mapping):
        value = obj.get(key[0], None)
    if value is None and hasattr(obj, key[0]):
        value = getattr(obj, key[0])
    if len(key) > 1 and value is not None:
        value = get_value(value, key[1])
    return value


class RelationalBundle(Generic[V], Bundle[V]):
    def __init__(self, items: Mapping[str, V], attributes: List[str], **kwargs) -> None:

        self._dataframe = DataFrame(columns=attributes, index=items.keys())
        for k, v in items.items():
            self._dataframe.loc[k] = Series(
                dict((a, get_value(v, a)) for a in attributes)
            )

        super().__init__(items, attributes=attributes, **kwargs)

    def __repr__(self) -> str:
        return self._dataframe.__repr__()

    def _repr_html_(self) -> Optional[str]:
        return self._dataframe._repr_html_()

    @property
    def dataframe(self) -> DataFrame:
        return self._dataframe

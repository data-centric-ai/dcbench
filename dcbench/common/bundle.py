from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, TypeVar, Union

from pandas import DataFrame, Series

BUNDLE_KEY_SEPARATOR = "/"
V = TypeVar("V")


class Bundle(Generic[V], Mapping[str, Union[V, "Bundle"]]):
    def __init__(self, items: Mapping[str, V] = {}, **kwargs) -> None:
        self._items: Mapping[str, Optional[V]] = items
        aux_items: Dict[str, Dict[str, Any]] = dict()
        for k, v in items.items():
            splits = k.split(BUNDLE_KEY_SEPARATOR, 1)
            if len(splits) > 1:
                aux_items.setdefault(splits[0], {})[splits[1]] = v
        self._aux_items: Dict[str, "Bundle[V]"] = dict(
            (k, type(self)(v, **kwargs)) for k, v in aux_items.items()
        )

        super().__init__()

    def __getitem__(self, k: str) -> Union[V, "Bundle[V]"]:
        result: Union[V, "Bundle[V]", None] = self._items.get(k, None)
        if result is None:
            result = self._aux_items.get(k)
        if result is None:
            raise KeyError()
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
    keyparts = key.split(".", maxsplit=1)
    if isinstance(obj, Mapping):
        value = obj.get(keyparts[0], None)
    if value is None and hasattr(obj, keyparts[0]):
        value = getattr(obj, keyparts[0])
    if len(keyparts) > 1 and value is not None:
        value = get_value(value, keyparts[1])
    return value


class RelationalBundle(Generic[V], Bundle[V]):
    def __init__(
        self, items: Mapping[str, V] = {}, attributes: List[str] = [], **kwargs
    ) -> None:

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

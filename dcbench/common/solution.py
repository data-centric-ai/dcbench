import json
from typing import Iterator, Mapping

from pandas import Series

from .artifact import ArtifactContainer


class Result(Mapping):
    def __init__(self, source: Mapping[str, float]) -> None:
        self._store = dict(source)
        super().__init__()

    def __getitem__(self, k: str) -> float:
        return self._store.get(k)

    def __iter__(self) -> Iterator[str]:
        return self._store.__iter__()

    def __len__(self) -> int:
        return len(self._store)

    def save(self, path: str) -> None:
        json.dump(self._store, path)

    @staticmethod
    def load(path: str) -> "Result":
        source = json.load(path)
        return Result(source)

    def __repr__(self) -> str:
        return Series(self).__repr__()


class Solution(ArtifactContainer):
    container_type: str = "solution"

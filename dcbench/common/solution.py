import json
import os
import random
import string
from typing import Any, Iterator, List, Mapping, Optional

from pandas import Series

import dcbench.constants as constants
from dcbench import config

from ..common import Problem
from .artefact import ArtefactContainer
from .download_utils import download_and_extract_archive


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


class Solution(ArtefactContainer):
    container_type: str = "solution"

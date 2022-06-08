import json
from typing import Iterator, Mapping

from pandas import Series

from .artifact_container import ArtifactContainer


class Solution(ArtifactContainer):
    container_type: str = "solution"

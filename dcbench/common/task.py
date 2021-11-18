import functools
import os
from dataclasses import dataclass
from typing import Sequence
from urllib.request import urlretrieve

import yaml
from meerkat.tools.lazy_loader import LazyLoader
from tqdm import tqdm

from dcbench.common.table import RowMixin, Table
from dcbench.config import config

from .artifact import ArtifactContainer

storage = LazyLoader("google.cloud.storage")


@dataclass
class Task(RowMixin):

    task_id: str
    name: str
    summary: str
    problem_class: type
    solution_class: type
    baselines: Table = Table([])

    def __post_init__(self):
        super().__init__(
            id=self.task_id, attributes={"name": self.name, "summary": self.summary}
        )

    @property
    def problems_path(self):
        return os.path.join(self.task_id, "problems.yaml")

    @property
    def local_problems_path(self):
        return os.path.join(config.local_dir, self.problems_path)

    @property
    def remote_problems_url(self):
        return os.path.join(config.public_remote_url, self.problems_path)

    def write_problems(self, containers: Sequence[ArtifactContainer]):
        ids = []
        for container in containers:
            assert isinstance(container, self.problem_class)
            ids.append(container.id)

        if len(set(ids)) != len(ids):
            raise ValueError(
                "Duplicate container ids in the containers passed to `write_problems`."
            )
        os.makedirs(os.path.dirname(self.local_problems_path), exist_ok=True)
        yaml.dump(containers, open(self.local_problems_path, "w"))

    def upload_problems(self, include_artifacts: bool = False):
        client = storage.Client()
        bucket = client.get_bucket(config.public_bucket_name)
        for container in tqdm(self.problems.values()):
            assert isinstance(container, self.problem_class)
            if include_artifacts:
                container.upload(bucket=bucket, force=True)
        blob = bucket.blob(self.problems_path)
        blob.upload_from_filename(self.local_problems_path)

    def download_problems(self, include_artifacts: bool = False):
        os.makedirs(os.path.dirname(self.local_problems_path), exist_ok=True)
        urlretrieve(self.remote_problems_url, self.local_problems_path)

        for container in self.problems.values():
            assert isinstance(container, self.problem_class)
            if include_artifacts:
                container.upload()

    @property
    @functools.lru_cache()
    def problems(self):
        if not os.path.exists(self.local_problems_path):
            self.download_problems()
        problems = yaml.load(open(self.local_problems_path), Loader=yaml.FullLoader)
        return Table(problems)

    def __repr__(self):
        return f'Task(task_id="{self.task_id}", name="{self.name}")'

    def __hash__(self):
        # necessary for lru cache
        return hash(repr(self))

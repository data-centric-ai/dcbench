from calendar import LocaleTextCalendar
import functools
import os
from dataclasses import dataclass
from typing import List
from urllib.request import urlretrieve
import warnings
import datetime
import uuid

import yaml
from meerkat.tools.lazy_loader import LazyLoader
from tqdm import tqdm

from dcbench.common.problem import ProblemTable
from dcbench.common.table import RowMixin, Table
from dcbench.config import config

from .artifact_container import ArtifactContainer
from .solution import Solution
from .problem import Problem

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

    def write_problems(self, containers: List[Problem], append: bool = True):
        ids = []
        for container in containers:
            assert isinstance(container, self.problem_class)
            ids.append(container.id)

        if len(set(ids)) != len(ids):
            raise ValueError(
                "Duplicate container ids in the containers passed to `write_problems`."
            )

        if append:
            for id, problem in self.problems.items():
                if id not in ids:
                    containers.append(problem)

        os.makedirs(os.path.dirname(self.local_problems_path), exist_ok=True)
        yaml.dump(containers, open(self.local_problems_path, "w"))
        self._load_problems.cache_clear()

    def solution_set_path(self, set_id: str = None):
        if set_id is None:
            # create unique id with today's date formatted like YY-MM-DD and a hash
            set_id = f"{datetime.date.today():%y-%m-%d}-{str(uuid.uuid4())[:8]}"
        return os.path.join(self.task_id, f"solution_sets/{set_id}/solutions.yaml")

    def local_solution_set_path(self, set_id: str = None):
        path = self.solution_set_path(set_id=set_id)
        return os.path.join(config.local_dir, path)


    def upload_problems(self, include_artifacts: bool = False, force: bool = True):
        """
        Uploads the problems to the remote storage.

        Args:
            include_artifacts (bool): If True, also uploads the artifacts of the
                problems.
            force (bool): If True, if the problem overwrites the remote problems.
                Defaults to True.
                .. warning::

                    It is somewhat dangerous to set `force=False`, as this could lead
                    to remote and local problems being out of sync.
        """
        client = storage.Client()
        bucket = client.get_bucket(config.public_bucket_name)

        local_problems = self.problems
        if not force and False:
            temp_fp, _ = urlretrieve(self.remote_problems_url)
            remote_problems_ids = [
                problem.id
                for problem in yaml.load(open(temp_fp), Loader=yaml.FullLoader)
            ]
            for problem_id in list(local_problems.keys()):
                if problem_id in remote_problems_ids:
                    warnings.warn(
                        f"Skipping problem {problem_id} because it is already uploaded."
                    )
                    del local_problems._data[problem_id]

        for container in tqdm(local_problems.values()):
            assert isinstance(container, self.problem_class)
            if include_artifacts:
                container.upload(bucket=bucket, force=force)
        blob = bucket.blob(self.problems_path)
        blob.upload_from_filename(self.local_problems_path)

    def download_problems(self, include_artifacts: bool = False):
        os.makedirs(os.path.dirname(self.local_problems_path), exist_ok=True)
        # TODO: figure out issue with caching on this call to urlretrieve
        urlretrieve(self.remote_problems_url, self.local_problems_path)
        self._load_problems.cache_clear()

        for container in self.problems.values():
            assert isinstance(container, self.problem_class)
            if include_artifacts:
                container.download()

    @functools.lru_cache()
    def _load_problems(self):
        if not os.path.exists(self.local_problems_path):
            self.download_problems()
        problems = yaml.load(open(self.local_problems_path), Loader=yaml.FullLoader)
        return ProblemTable(problems)

    @property
    def problems(self):
        return self._load_problems()
    

    @property
    def solution_sets(self):
        return list(os.listdir(
            os.path.join(config.local_dir, self.task_id, "solution_sets")
        ))

    def __repr__(self):
        return f'Task(task_id="{self.task_id}", name="{self.name}")'

    def __hash__(self):
        # necessary for lru cache
        return hash(repr(self))
